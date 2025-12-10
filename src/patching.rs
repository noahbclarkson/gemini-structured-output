use std::sync::Arc;
use tokio::time::{sleep, Duration};

use gemini_rust::{Content, Gemini, GenerationConfig, Message, Role};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use tracing::{debug, info, instrument, trace, warn};

use crate::{
    client::FallbackStrategy,
    error::{Result, StructuredError},
    models::{RefinementAttempt, RefinementOutcome},
    schema::{compile_validator, GeminiStructured, StructuredValidator},
};

/// Configuration for the refinement engine.
#[derive(Clone, Debug)]
pub struct RefinementConfig {
    /// Maximum number of retry attempts (default: 3)
    pub max_retries: usize,
    /// Temperature for patch generation (default: 0.0)
    pub temperature: f32,
    /// Strategy for handling patch application
    pub patch_strategy: PatchStrategy,
    /// Strategy for handling arrays in patches
    pub array_strategy: ArrayPatchStrategy,
    /// Network retries for transient generation failures (e.g., 503/429).
    pub network_retries: usize,
    /// Strategy for model escalation when primary model fails repeatedly.
    pub fallback_strategy: FallbackStrategy,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            temperature: 0.0,
            patch_strategy: PatchStrategy::PartialApply,
            array_strategy: ArrayPatchStrategy::ReplaceWhole,
            network_retries: 3,
            fallback_strategy: FallbackStrategy::default(),
        }
    }
}

/// Strategy for handling array modifications in patches.
#[derive(Clone, Debug, Default)]
pub enum ArrayPatchStrategy {
    /// Replace the entire array when any element changes (safest, default)
    #[default]
    ReplaceWhole,
    /// Apply patches as-is (may fail with index shifts)
    Direct,
    /// Reorder patches to apply removals in reverse order
    ReorderRemovals,
}

/// Strategy for applying patch operations.
#[derive(Clone, Debug, Default)]
pub enum PatchStrategy {
    /// Apply patches individually; keep successful changes even if some fail.
    #[default]
    PartialApply,
    /// Apply as a single transaction; any failure discards all operations.
    Atomic,
}

/// Runs an instruction-driven JSON Patch refinement loop.
#[derive(Clone)]
pub struct RefinementEngine {
    primary_client: Arc<Gemini>,
    fallback_client: Option<Arc<Gemini>>,
    config: RefinementConfig,
}

impl RefinementEngine {
    pub fn new(primary_client: Arc<Gemini>, fallback_client: Option<Arc<Gemini>>) -> Self {
        Self {
            primary_client,
            fallback_client,
            config: RefinementConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RefinementConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.config.max_retries = max_retries.max(1);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn with_array_strategy(mut self, strategy: ArrayPatchStrategy) -> Self {
        self.config.array_strategy = strategy;
        self
    }

    /// Refine an existing value into a new one using JSON Patch.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn refine<T>(&self, current: &T, instruction: &str) -> Result<RefinementOutcome<T>>
    where
        T: GeminiStructured + StructuredValidator + Serialize + DeserializeOwned + Clone,
    {
        let schema = T::gemini_schema();
        let validator = compile_validator::<T>()?;
        let mut working = serde_json::to_value(current)?;
        let mut attempts = Vec::new();
        let mut conversation: Vec<Message> = Vec::new();
        let mut escalated = false;

        let system_prompt = self.build_system_prompt();

        debug!(
            "Starting refinement loop with {:?}",
            self.config.array_strategy
        );

        for attempt_idx in 1..=self.config.max_retries {
            // Determine which client to use based on escalation strategy
            let active_client = self.select_client(attempt_idx, &mut escalated);

            let prompt = format!(
                "Current JSON:\n{}\n\nTarget schema:\n{}\n\nInstruction:\n{}\n\nReturn a JSON Patch array:",
                serde_json::to_string_pretty(&working)?,
                serde_json::to_string_pretty(&schema)?,
                instruction
            );

            let response = {
                let mut last_err: Option<StructuredError> = None;
                let mut captured: Option<gemini_rust::GenerationResponse> = None;

                for net_try in 0..=self.config.network_retries {
                    let mut builder = active_client
                        .generate_content()
                        .with_system_instruction(&system_prompt)
                        .with_generation_config(GenerationConfig {
                            response_mime_type: Some("application/json".to_string()),
                            temperature: Some(self.config.temperature),
                            ..Default::default()
                        });

                    for msg in &conversation {
                        builder = builder.with_message(msg.clone());
                    }

                    builder = builder.with_message(Message {
                        role: Role::User,
                        content: Content::text(prompt.clone()).with_role(Role::User),
                    });

                    match builder.execute().await {
                        Ok(resp) => {
                            captured = Some(resp);
                            last_err = None;
                            break;
                        }
                        Err(err) => {
                            let structured = StructuredError::Gemini(err);
                            if structured.is_retryable() && net_try < self.config.network_retries {
                                let delay_ms = 200 * 2_u64.pow(net_try as u32);
                                warn!(
                                    attempt = attempt_idx,
                                    network_try = net_try + 1,
                                    "Transient error ({}). Retrying after {}ms",
                                    structured,
                                    delay_ms
                                );
                                sleep(Duration::from_millis(delay_ms)).await;
                                last_err = Some(structured);
                                continue;
                            } else {
                                last_err = Some(structured);
                                break;
                            }
                        }
                    }
                }

                captured.ok_or_else(|| {
                    last_err.unwrap_or_else(|| StructuredError::RefinementExhausted {
                        retries: self.config.max_retries,
                        last_error: "refinement request failed".to_string(),
                    })
                })?
            };

            let patch_text = response.text();

            trace!(patch = %patch_text, "Received patch from model");

            conversation.push(Message::model(patch_text.clone()));

            let cleaned_patch = clean_patch_text(&patch_text);
            let mut patch: json_patch::Patch = match serde_json::from_str(cleaned_patch) {
                Ok(p) => p,
                Err(e) => {
                    let msg = format!(
                        "Model response was not valid JSON Patch: {e}; body={cleaned_patch}"
                    );
                    warn!(attempt = attempt_idx, error = %msg, "Invalid JSON Patch from model");
                    attempts.push(RefinementAttempt::failure(patch_text.clone(), msg.clone()));
                    conversation.push(Message::user(format!(
                        "The patch could not be parsed: {msg}. Return only a valid JSON Patch array."
                    )));
                    continue;
                }
            };

            if matches!(
                self.config.array_strategy,
                ArrayPatchStrategy::ReorderRemovals
            ) {
                patch = self.reorder_removals(patch);
            }

            let (next_value, patch_errors) = self.apply_patches(&working, &patch);

            if !patch_errors.is_empty() {
                let msg = patch_errors.join("; ");
                warn!(
                    attempt = attempt_idx,
                    errors = %msg,
                    patch_text = %patch_text,
                    "Patch application had failures"
                );
                attempts.push(RefinementAttempt::failure(patch_text.clone(), msg.clone()));
                conversation.push(Message::user(format!(
                    "Some patch operations failed: {msg}. Fix the remaining issues."
                )));

                if matches!(self.config.patch_strategy, PatchStrategy::PartialApply) {
                    working = next_value;
                }
                continue;
            }

            let candidate = next_value;

            if !validator.is_valid(&candidate) && !validator.is_valid(&candidate) {
                let msg = validator
                    .iter_errors(&candidate)
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");

                warn!(
                    attempt = attempt_idx,
                    error = %msg,
                    patch_text = %patch_text,
                    candidate_json = %candidate,
                    "Patch resulted in invalid JSON schema"
                );

                attempts.push(RefinementAttempt::failure(patch_text.clone(), msg.clone()));
                conversation.push(Message::user(format!(
                    "Patch failed validation: {msg}. Return a corrected JSON Patch."
                )));
                working = candidate;
                continue;
            }

            let value: T = serde_json::from_value(candidate.clone())?;
            if let Some(logic_err) = value.validate() {
                warn!(
                    attempt = attempt_idx,
                    error = %logic_err,
                    "Patch passed schema but failed logic validation"
                );

                attempts.push(RefinementAttempt::failure(
                    patch_text.clone(),
                    logic_err.clone(),
                ));
                conversation.push(Message::user(format!(
                    "JSON is valid, but logic failed: {logic_err}. Fix the data."
                )));
                working = serde_json::to_value(&value)?;
                continue;
            }

            debug!("Refinement successful on attempt {}", attempt_idx);
            attempts.push(RefinementAttempt::success(patch_text));
            return Ok(RefinementOutcome::new(value, attempts));
        }

        Err(StructuredError::RefinementExhausted {
            retries: self.config.max_retries,
            last_error: attempts
                .last()
                .and_then(|a| a.error.clone())
                .unwrap_or_else(|| "unknown error".to_string()),
        })
    }

    fn apply_patches(&self, original: &Value, patch: &json_patch::Patch) -> (Value, Vec<String>) {
        match self.config.patch_strategy {
            PatchStrategy::Atomic => {
                let mut doc = original.clone();
                match json_patch::patch(&mut doc, patch) {
                    Ok(_) => (doc, vec![]),
                    Err(e) => (original.clone(), vec![format!("Atomic failure: {}", e)]),
                }
            }
            PatchStrategy::PartialApply => {
                let mut doc = original.clone();
                let mut errors = Vec::new();

                for op in &patch.0 {
                    let mut temp = doc.clone();
                    let single = json_patch::Patch(vec![op.clone()]);
                    match json_patch::patch(&mut temp, &single) {
                        Ok(_) => doc = temp,
                        Err(e) => errors.push(format!("Op failed (path: {}): {}", op_path(op), e)),
                    }
                }

                (doc, errors)
            }
        }
    }

    fn build_system_prompt(&self) -> String {
        let base = "You are a JSON Patch generator. Given the current JSON value and the target schema, \
                    return ONLY a valid RFC6902 JSON Patch array that transforms the current value to \
                    satisfy the instruction and schema. Do not wrap in code fences or prose.";

        let array_guidance = match self.config.array_strategy {
            ArrayPatchStrategy::ReplaceWhole => {
                "\n\nIMPORTANT: When modifying arrays, prefer using a single 'replace' operation \
                 on the entire array (e.g., {\"op\": \"replace\", \"path\": \"/items\", \"value\": [...]}) \
                 rather than individual add/remove operations on array indices. This prevents index \
                 shift issues when patches are applied sequentially."
            }
            ArrayPatchStrategy::ReorderRemovals => {
                "\n\nWhen removing multiple array elements, list removals in reverse index order \
                 (highest index first) to prevent index shift issues."
            }
            ArrayPatchStrategy::Direct => "",
        };

        format!("{}{}", base, array_guidance)
    }

    /// Select the appropriate client based on the escalation strategy.
    fn select_client(&self, attempt_idx: usize, escalated: &mut bool) -> &Arc<Gemini> {
        match &self.config.fallback_strategy {
            FallbackStrategy::Escalate {
                after_attempts,
                target: _,
            } if attempt_idx > *after_attempts && self.fallback_client.is_some() => {
                if !*escalated {
                    info!(
                        attempt = attempt_idx,
                        after_attempts = after_attempts,
                        "Escalating refinement to fallback model"
                    );
                    *escalated = true;
                }
                self.fallback_client.as_ref().unwrap()
            }
            _ => &self.primary_client,
        }
    }

    /// Reorder removal operations to process higher indices first.
    fn reorder_removals(&self, patch: json_patch::Patch) -> json_patch::Patch {
        let mut ops: Vec<json_patch::PatchOperation> = patch.0.into_iter().collect();

        // Separate removals from other operations
        let (mut removals, others): (Vec<_>, Vec<_>) = ops
            .drain(..)
            .partition(|op| matches!(op, json_patch::PatchOperation::Remove(_)));

        // Sort removals by path index in descending order
        removals.sort_by(|a, b| {
            let idx_a = extract_array_index(a);
            let idx_b = extract_array_index(b);
            idx_b.cmp(&idx_a)
        });

        // Rebuild: other ops first, then sorted removals
        let mut result = others;
        result.extend(removals);
        json_patch::Patch(result)
    }
}

fn clean_patch_text(patch_text: &str) -> &str {
    let trimmed = patch_text.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            return &trimmed[start..=end];
        }
    }
    trimmed
}

fn op_path(op: &json_patch::PatchOperation) -> String {
    use json_patch::PatchOperation;

    match op {
        PatchOperation::Add(add_op) => add_op.path.to_string(),
        PatchOperation::Remove(remove_op) => remove_op.path.to_string(),
        PatchOperation::Replace(replace_op) => replace_op.path.to_string(),
        PatchOperation::Move(move_op) => move_op.path.to_string(),
        PatchOperation::Copy(copy_op) => copy_op.path.to_string(),
        PatchOperation::Test(test_op) => test_op.path.to_string(),
    }
}

/// Extract array index from a patch operation's path, if present.
fn extract_array_index(op: &json_patch::PatchOperation) -> Option<usize> {
    use json_patch::PatchOperation;

    let path_str = match op {
        PatchOperation::Add(add_op) => add_op.path.to_string(),
        PatchOperation::Remove(remove_op) => remove_op.path.to_string(),
        PatchOperation::Replace(replace_op) => replace_op.path.to_string(),
        PatchOperation::Move(move_op) => move_op.path.to_string(),
        PatchOperation::Copy(copy_op) => copy_op.path.to_string(),
        PatchOperation::Test(test_op) => test_op.path.to_string(),
    };

    // Get the last segment of the path
    path_str
        .rsplit('/')
        .next()
        .and_then(|s| s.parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct TestItem {
        id: i32,
        name: String,
        value: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct TestContainer {
        items: Vec<TestItem>,
        total: f64,
    }

    #[test]
    fn test_reorder_removals() {
        let patch_json = r#"[
            {"op": "remove", "path": "/items/0"},
            {"op": "remove", "path": "/items/2"},
            {"op": "remove", "path": "/items/1"}
        ]"#;

        let patch: json_patch::Patch = serde_json::from_str(patch_json).unwrap();
        let engine = RefinementEngine::new(Arc::new(Gemini::new("test").unwrap()), None);

        let reordered = engine.reorder_removals(patch);

        // Should be reordered to: 2, 1, 0
        let ops: Vec<_> = reordered.0.iter().collect();
        assert_eq!(ops.len(), 3);

        let indices: Vec<Option<usize>> = ops.iter().map(|op| extract_array_index(op)).collect();
        assert_eq!(indices, vec![Some(2), Some(1), Some(0)]);
    }

    #[test]
    fn test_apply_patch_with_replace_whole_array() {
        let original = serde_json::json!({
            "items": [
                {"id": 1, "name": "A", "value": 10.0},
                {"id": 2, "name": "B", "value": 20.0}
            ],
            "total": 30.0
        });

        let patch_json = r#"[
            {"op": "replace", "path": "/items", "value": [
                {"id": 1, "name": "A", "value": 15.0},
                {"id": 2, "name": "B", "value": 25.0},
                {"id": 3, "name": "C", "value": 10.0}
            ]},
            {"op": "replace", "path": "/total", "value": 50.0}
        ]"#;

        let engine = RefinementEngine::new(Arc::new(Gemini::new("test").unwrap()), None);
        let patch: json_patch::Patch = serde_json::from_str(patch_json).unwrap();
        let (result, errors) = engine.apply_patches(&original, &patch);
        assert!(errors.is_empty(), "Expected no patch errors: {:?}", errors);

        let container: TestContainer = serde_json::from_value(result).unwrap();
        assert_eq!(container.items.len(), 3);
        assert_eq!(container.total, 50.0);
        assert_eq!(container.items[2].name, "C");
    }

    #[test]
    fn test_extract_array_index() {
        let op = json_patch::PatchOperation::Remove(json_patch::RemoveOperation {
            path: "/items/5".parse().unwrap(),
        });
        assert_eq!(extract_array_index(&op), Some(5));

        let op = json_patch::PatchOperation::Replace(json_patch::ReplaceOperation {
            path: "/name".parse().unwrap(),
            value: serde_json::json!("test"),
        });
        assert_eq!(extract_array_index(&op), None);
    }
}
