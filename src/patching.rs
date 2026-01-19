use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

use gemini_rust::{Content, FileHandle, Gemini, GenerationConfig, Message, Part, Role};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, instrument, trace, warn};

use crate::{
    client::FallbackStrategy,
    error::{Result, StructuredError},
    files::FileManager,
    generator::TextGenerator,
    models::{RefinementAttempt, RefinementOutcome},
    schema::{
        clean_schema_for_gemini, coerce_enum_strings, compile_validator, prune_null_fields,
        recover_internally_tagged_enums, strip_x_fields, unflatten_externally_tagged_enums,
        warn_if_schema_too_deep, GeminiStructured, StructuredValidator,
    },
    StructuredClient,
};

/// Closure type for generating dynamic context per refinement iteration.
pub type ContextGenerator<T> = Box<dyn Fn(&T) -> String + Send + Sync>;
/// Closure type for external/context-aware validation.
pub type CustomValidator<T> = Box<dyn Fn(&T) -> Option<String> + Send + Sync>;
/// Boxed future helper for async validators.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
/// Closure type for async context-aware validation.
pub type AsyncCustomValidator<T> =
    Box<dyn Fn(&T) -> BoxFuture<'static, Option<String>> + Send + Sync>;

/// Strategy for handling validation failures during refinement.
#[derive(Clone, Debug, Default)]
pub enum ValidationFailureStrategy {
    /// Keep the invalid state and ask the model to fix it.
    #[default]
    IterateForward,
    /// Revert to the last valid state and ask the model to try a different approach.
    Rollback,
}

/// Schema definition for a JSON Patch operation to ensure strict structured output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "lowercase")]
enum PatchOperationSchema {
    Add { path: String, value: Value },
    Remove { path: String },
    Replace { path: String, value: Value },
    Move { from: String, path: String },
    Copy { from: String, path: String },
    Test { path: String, value: Value },
}

/// Wrapper for the patch array to satisfy Gemini's preference for root objects.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct PatchResult {
    patch: Vec<PatchOperationSchema>,
}

/// Builder for configuring and executing refinement with optional documents and dynamic context.
pub struct RefinementRequest<'a, T> {
    client: &'a StructuredClient,
    current: T,
    instruction: String,
    files: Vec<FileHandle>,
    context_generator: Option<ContextGenerator<T>>,
    custom_validator: Option<CustomValidator<T>>,
    async_custom_validator: Option<AsyncCustomValidator<T>>,
}

impl<'a, T> RefinementRequest<'a, T>
where
    T: GeminiStructured
        + StructuredValidator
        + Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static,
{
    pub fn new(client: &'a StructuredClient, current: T, instruction: String) -> Self {
        Self {
            client,
            current,
            instruction,
            files: Vec::new(),
            context_generator: None,
            custom_validator: None,
            async_custom_validator: None,
        }
    }

    /// Attach file handles (PDFs/images) to the refinement context.
    pub fn with_documents(mut self, documents: Vec<FileHandle>) -> Self {
        self.files = documents;
        self
    }

    /// Register a context-aware validator executed after schema and internal validation.
    ///
    /// Return `Some(error_message)` to signal invalid data; the message is fed back to the model.
    pub fn with_validator<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> Option<String> + Send + Sync + 'static,
    {
        self.custom_validator = Some(Box::new(f));
        self
    }

    /// Register an asynchronous context-aware validator for heavy checks.
    pub fn with_async_validator<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<String>> + Send + 'static,
    {
        let f = Arc::new(f);
        self.async_custom_validator = Some(Box::new(move |t: &T| {
            let owned = t.clone();
            let func = Arc::clone(&f);
            Box::pin(async move { func(&owned).await })
        }));
        self
    }

    /// Inject dynamic context built from the current value on each iteration.
    pub fn with_context_generator<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> String + Send + Sync + 'static,
    {
        self.context_generator = Some(Box::new(f));
        self
    }

    /// Execute the refinement loop.
    pub async fn execute(self) -> Result<RefinementOutcome<T>> {
        let mut initial_history = Vec::new();

        if !self.files.is_empty() {
            let mut parts: Vec<Part> = Vec::new();
            parts.push(Part::Text {
                text: "Reference documents for this refinement task:".to_string(),
                thought: None,
                thought_signature: None,
            });

            for file in self.files {
                let part = FileManager::as_part(&file)?;
                parts.push(part);
            }

            let content = Content {
                parts: Some(parts),
                role: Some(Role::User),
            };
            initial_history.push(Message {
                role: Role::User,
                content,
            });
        }

        self.client
            .refiner()
            .execute_refinement(
                self.current,
                self.instruction,
                initial_history,
                self.context_generator.as_ref(),
                self.custom_validator.as_ref(),
                self.async_custom_validator.as_ref(),
            )
            .await
    }
}

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
    /// Strategy for handling validation failures (iterate or rollback).
    pub validation_failure_strategy: ValidationFailureStrategy,
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
            validation_failure_strategy: ValidationFailureStrategy::default(),
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
///
/// The engine supports two modes of operation:
/// 1. **Conversational mode** (default): Uses the Gemini client directly with
///    conversation history for multi-turn patch generation.
/// 2. **Generator mode**: Uses a `TextGenerator` trait object for simpler,
///    single-turn patch generation. This mode is useful for testing or
///    when using alternative backends.
#[derive(Clone)]
pub struct RefinementEngine {
    primary_client: Arc<Gemini>,
    fallback_client: Option<Arc<Gemini>>,
    primary_generator: Option<Arc<dyn TextGenerator>>,
    fallback_generator: Option<Arc<dyn TextGenerator>>,
    config: RefinementConfig,
}

impl RefinementEngine {
    /// Create a new refinement engine with Gemini clients.
    ///
    /// This enables conversational refinement with full conversation history support.
    pub fn new(primary_client: Arc<Gemini>, fallback_client: Option<Arc<Gemini>>) -> Self {
        Self {
            primary_client,
            fallback_client,
            primary_generator: None,
            fallback_generator: None,
            config: RefinementConfig::default(),
        }
    }

    /// Create a new refinement engine from TextGenerator trait objects.
    ///
    /// This enables using alternative backends or mock generators for testing.
    /// Note: This mode uses single-turn generation without conversation history.
    pub fn from_generators(
        primary: Arc<dyn TextGenerator>,
        fallback: Option<Arc<dyn TextGenerator>>,
    ) -> Self {
        Self {
            primary_client: Arc::new(Gemini::new("unused").expect("Unused client")),
            fallback_client: None,
            primary_generator: Some(primary),
            fallback_generator: fallback,
            config: RefinementConfig::default(),
        }
    }

    /// Get the primary generator, if one was configured.
    pub fn generator(&self) -> Option<&Arc<dyn TextGenerator>> {
        self.primary_generator.as_ref()
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

    /// Refine an existing value into a new one using JSON Patch (compat wrapper).
    pub async fn refine<T>(&self, current: &T, instruction: &str) -> Result<RefinementOutcome<T>>
    where
        T: GeminiStructured + StructuredValidator + Serialize + DeserializeOwned + Clone,
    {
        self.execute_refinement(
            current.clone(),
            instruction.to_string(),
            Vec::new(),
            None,
            None,
            None,
        )
        .await
    }

    /// Core refinement runner with optional initial history and dynamic context.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub(crate) async fn execute_refinement<T>(
        &self,
        current: T,
        instruction: String,
        initial_history: Vec<Message>,
        context_generator: Option<&ContextGenerator<T>>,
        custom_validator: Option<&CustomValidator<T>>,
        async_custom_validator: Option<&AsyncCustomValidator<T>>,
    ) -> Result<RefinementOutcome<T>>
    where
        T: GeminiStructured + StructuredValidator + Serialize + DeserializeOwned + Clone,
    {
        let schema = T::gemini_schema();
        let validator = compile_validator::<T>()?;
        let mut working = serde_json::to_value(&current)?;
        let original_instruction = instruction.clone();
        let mut attempts = Vec::new();
        let mut conversation: Vec<Message> = initial_history;
        let mut escalated = false;
        let use_generator = self.uses_generators();

        let system_prompt = self.build_system_prompt();
        let mut patch_schema = PatchResult::gemini_schema();
        clean_schema_for_gemini(&mut patch_schema);
        strip_x_fields(&mut patch_schema);
        warn_if_schema_too_deep(&patch_schema, crate::schema::STRICT_SCHEMA_DEPTH_LIMIT);

        debug!(
            "Starting refinement loop with {:?}",
            self.config.array_strategy
        );

        for attempt_idx in 1..=self.config.max_retries {
            let previous_valid = working.clone();
            let current_struct: T = serde_json::from_value(working.clone())?;
            let dynamic_context = context_generator
                .map(|gen| gen(&current_struct))
                .unwrap_or_default();

            let prompt = format!(
                "Current JSON:\n{}\n\nTarget schema:\n{}\n\n{}Instruction:\n{}\n\nReturn a JSON object with a 'patch' array:",
                serde_json::to_string_pretty(&working)?,
                serde_json::to_string_pretty(&schema)?,
                if dynamic_context.is_empty() {
                    String::new()
                } else {
                    format!("Additional context:\n{}\n\n", dynamic_context)
                },
                instruction
            );

            let patch_text: String = if use_generator {
                let generator = self
                    .select_generator(attempt_idx, &mut escalated)
                    .ok_or_else(|| {
                        StructuredError::Config("No generator configured".to_string())
                    })?;

                generator
                    .generate_text(
                        Some(&system_prompt),
                        &prompt,
                        GenerationConfig {
                            response_mime_type: Some("application/json".to_string()),
                            response_json_schema: Some(patch_schema.clone()),
                            response_schema: None,
                            temperature: Some(self.config.temperature),
                            ..Default::default()
                        },
                    )
                    .await?
            } else {
                // Determine which client to use based on escalation strategy
                let active_client = self.select_client(attempt_idx, &mut escalated);

                let response = {
                    let mut last_err: Option<StructuredError> = None;
                    let mut captured: Option<gemini_rust::GenerationResponse> = None;

                    for net_try in 0..=self.config.network_retries {
                        let mut builder = active_client
                            .generate_content()
                            .with_system_instruction(&system_prompt)
                            .with_generation_config(GenerationConfig {
                                response_mime_type: Some("application/json".to_string()),
                                response_json_schema: Some(patch_schema.clone()),
                                response_schema: None,
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
                                if structured.is_retryable()
                                    && net_try < self.config.network_retries
                                {
                                    // Use API-provided retry delay if available, otherwise exponential backoff
                                    let delay = structured
                                        .retry_delay()
                                        .map(Duration::from_secs)
                                        .unwrap_or_else(|| {
                                            Duration::from_millis(200 * 2_u64.pow(net_try as u32))
                                        });
                                    warn!(
                                        attempt = attempt_idx,
                                        network_try = net_try + 1,
                                        "Transient error ({}). Retrying after {:?}",
                                        structured,
                                        delay
                                    );
                                    sleep(delay).await;
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

                patch_text
            };

            let cleaned_patch = clean_patch_text(&patch_text);
            let patch_result: PatchResult = match serde_json::from_str(cleaned_patch) {
                Ok(p) => p,
                Err(e) => {
                    if let Ok(raw_ops) =
                        serde_json::from_str::<Vec<PatchOperationSchema>>(cleaned_patch)
                    {
                        PatchResult { patch: raw_ops }
                    } else {
                        let msg = format!(
                            "Model response was not valid JSON Patch: {e}; body={cleaned_patch}"
                        );
                        warn!(attempt = attempt_idx, error = %msg, "Invalid JSON Patch from model");
                        attempts.push(RefinementAttempt::failure(patch_text.clone(), msg.clone()));
                        conversation.push(Message::user(format!(
                            "The patch could not be parsed: {msg}. Return a JSON object {{\"patch\": [...]}}.\n\n\
                             REMINDER - Original Instruction: {original_instruction}\n\
                             Fix the errors while ensuring the original instruction is still met."
                        )));
                        continue;
                    }
                }
            };

            let ops_value = serde_json::to_value(patch_result.patch)?;
            let mut patch: json_patch::Patch = serde_json::from_value(ops_value)?;

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
                    "Some patch operations failed: {msg}.\n\n\
                     REMINDER - Original Instruction: {original_instruction}\n\
                     Fix the errors while ensuring the original instruction is still met."
                )));

                if matches!(self.config.patch_strategy, PatchStrategy::PartialApply) {
                    working = next_value;
                }
                continue;
            }

            let mut candidate = next_value;
            Self::normalize_candidate_for_schema(&mut candidate, &schema);

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
                    "Patch failed validation: {msg}.\n\n\
                     REMINDER - Original Instruction: {original_instruction}\n\
                     Return a corrected JSON Patch while keeping the instruction in mind."
                )));
                match self.config.validation_failure_strategy {
                    ValidationFailureStrategy::IterateForward => working = candidate,
                    ValidationFailureStrategy::Rollback => {
                        working = previous_valid;
                        conversation.push(Message::user(
                            "The previous patch resulted in invalid data. Changes were reverted; try a different approach while honoring the original instruction.".to_string(),
                        ));
                    }
                }
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
                    "JSON is valid, but logic failed: {logic_err}.\n\n\
                     REMINDER - Original Instruction: {original_instruction}\n\
                     Fix the data while preserving the original intent."
                )));
                match self.config.validation_failure_strategy {
                    ValidationFailureStrategy::IterateForward => {
                        working = serde_json::to_value(&value)?;
                    }
                    ValidationFailureStrategy::Rollback => {
                        working = previous_valid;
                        conversation.push(Message::user(
                            "Logic validation failed. Reverted to last valid state; try a different patch that still meets the original instruction.".to_string(),
                        ));
                    }
                }
                continue;
            }

            if let Some(validator) = custom_validator {
                if let Some(ctx_err) = validator(&value) {
                    warn!(
                        attempt = attempt_idx,
                        error = %ctx_err,
                        "Context validation failed"
                    );

                    attempts.push(RefinementAttempt::failure(
                        patch_text.clone(),
                        ctx_err.clone(),
                    ));
                    conversation.push(Message::user(format!(
                        "The data structure is valid, but it violates external constraints: {ctx_err}.\n\n\
                         REMINDER - Original Instruction: {original_instruction}\n\
                         Please adjust the values to satisfy this constraint while honoring the instruction."
                    )));
                    match self.config.validation_failure_strategy {
                        ValidationFailureStrategy::IterateForward => {
                            working = serde_json::to_value(&value)?;
                        }
                        ValidationFailureStrategy::Rollback => {
                            working = previous_valid;
                            conversation.push(Message::user(
                                "Context validation failed. Reverted to last valid state; try a different approach that still satisfies the instruction.".to_string(),
                            ));
                        }
                    }
                    continue;
                }
            }

            if let Some(validator) = async_custom_validator {
                if let Some(async_err) = validator(&value).await {
                    warn!(
                        attempt = attempt_idx,
                        error = %async_err,
                        "Async context validation failed"
                    );

                    attempts.push(RefinementAttempt::failure(
                        patch_text.clone(),
                        async_err.clone(),
                    ));
                    conversation.push(Message::user(format!(
                        "The configuration structure is valid, but the simulation/async check failed: {async_err}.\n\n\
                         REMINDER - Original Instruction: {original_instruction}\n\
                         Please adjust the values to satisfy this constraint while preserving the instruction."
                    )));
                    match self.config.validation_failure_strategy {
                        ValidationFailureStrategy::IterateForward => {
                            working = serde_json::to_value(&value)?;
                        }
                        ValidationFailureStrategy::Rollback => {
                            working = previous_valid;
                            conversation.push(Message::user(
                                "Async validation failed. Reverted to last valid state; try a different approach that still satisfies the instruction.".to_string(),
                            ));
                        }
                    }
                    continue;
                }
            }

            debug!("Refinement successful on attempt {}", attempt_idx);
            attempts.push(RefinementAttempt::success(patch_text));
            let applied_patch = patch.clone();
            return Ok(RefinementOutcome::with_patch(
                value,
                attempts,
                Some(applied_patch),
            ));
        }

        Err(StructuredError::RefinementExhausted {
            retries: self.config.max_retries,
            last_error: attempts
                .last()
                .and_then(|a| a.error.clone())
                .unwrap_or_else(|| "unknown error".to_string()),
        })
    }

    fn normalize_candidate_for_schema(candidate: &mut Value, schema: &Value) {
        prune_null_fields(candidate);
        unflatten_externally_tagged_enums(candidate, schema);
        coerce_enum_strings(candidate, schema);
        recover_internally_tagged_enums(candidate, schema);
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
                    let path = op_path(op);

                    // Check if the parent path is valid (not null or missing)
                    // This prevents errors when LLM generates patches targeting paths through null values
                    if !is_parent_path_valid(&doc, &path) {
                        errors.push(format!(
                            "Skipped op (path: {}): parent path is null or missing - \
                             you may need to set the parent object first before setting nested fields",
                            path
                        ));
                        continue;
                    }

                    let mut temp = doc.clone();
                    let single = json_patch::Patch(vec![op.clone()]);
                    match json_patch::patch(&mut temp, &single) {
                        Ok(_) => doc = temp,
                        Err(e) => errors.push(format!("Op failed (path: {}): {}", path, e)),
                    }
                }

                (doc, errors)
            }
        }
    }

    fn build_system_prompt(&self) -> String {
        let base = "You are a JSON Patch generator. Given the current JSON value and the target schema, \
                    return a JSON object with a 'patch' key containing an array of valid RFC6902 \
                    operations that transforms the current value to satisfy the instruction and schema. \
                    Do not wrap in code fences or prose.";

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

    /// Select the appropriate generator based on the escalation strategy.
    ///
    /// Returns None if no generator was configured (use select_client instead).
    fn select_generator(
        &self,
        attempt_idx: usize,
        escalated: &mut bool,
    ) -> Option<&Arc<dyn TextGenerator>> {
        let primary = self.primary_generator.as_ref()?;

        match &self.config.fallback_strategy {
            FallbackStrategy::Escalate {
                after_attempts,
                target: _,
            } if attempt_idx > *after_attempts && self.fallback_generator.is_some() => {
                if !*escalated {
                    info!(
                        attempt = attempt_idx,
                        after_attempts = after_attempts,
                        "Escalating refinement to fallback generator"
                    );
                    *escalated = true;
                }
                self.fallback_generator.as_ref()
            }
            _ => Some(primary),
        }
    }

    /// Check if the engine is configured to use generators instead of clients.
    pub fn uses_generators(&self) -> bool {
        self.primary_generator.is_some()
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
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return &trimmed[start..=end];
        }
    }
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            return &trimmed[start..=end];
        }
    }
    trimmed
}

/// Check if the parent path of a JSON Pointer is valid and not null.
///
/// For a path like "/a/b/c/d", this checks that "/a/b/c" exists and is not null.
/// This prevents patch operations from failing when trying to traverse through null values.
fn is_parent_path_valid(doc: &Value, path: &str) -> bool {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    if segments.is_empty() {
        return true; // Root path is always valid
    }

    // Check all parent segments (all except the last one)
    let parent_segments = &segments[..segments.len() - 1];

    let mut current = doc;
    for segment in parent_segments {
        match current {
            Value::Object(map) => {
                match map.get(*segment) {
                    Some(Value::Null) => return false, // Can't traverse through null
                    Some(v) => current = v,
                    None => return false, // Path doesn't exist
                }
            }
            Value::Array(arr) => {
                if let Ok(idx) = segment.parse::<usize>() {
                    match arr.get(idx) {
                        Some(Value::Null) => return false, // Can't traverse through null
                        Some(v) => current = v,
                        None => return false, // Index out of bounds
                    }
                } else {
                    return false; // Invalid array index
                }
            }
            Value::Null => return false, // Can't traverse through null
            _ => return false, // Can't traverse into primitives
        }
    }

    true
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
    use serde_json::json;

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

    #[test]
    fn test_is_parent_path_valid() {
        // Valid paths
        let doc = serde_json::json!({
            "a": {
                "b": {
                    "c": "value"
                }
            }
        });
        assert!(is_parent_path_valid(&doc, "/a/b/c"));
        assert!(is_parent_path_valid(&doc, "/a/b"));
        assert!(is_parent_path_valid(&doc, "/a"));

        // Path through null
        let doc_with_null = serde_json::json!({
            "items": [
                { "source": null }
            ]
        });
        assert!(!is_parent_path_valid(
            &doc_with_null,
            "/items/0/source/document"
        ));

        // Valid path to null (parent exists)
        assert!(is_parent_path_valid(&doc_with_null, "/items/0/source"));

        // Missing path
        assert!(!is_parent_path_valid(&doc, "/a/x/y"));
    }

    #[test]
    fn test_apply_patch_skips_null_parent_paths() {
        let original = serde_json::json!({
            "items": [
                {
                    "name": "Item 1",
                    "metadata": null
                }
            ]
        });

        // Try to patch through a null value
        let patch_json = r#"[
            {"op": "replace", "path": "/items/0/metadata/description", "value": "test"},
            {"op": "replace", "path": "/items/0/name", "value": "Updated Item"}
        ]"#;

        let engine = RefinementEngine::new(Arc::new(Gemini::new("test").unwrap()), None);
        let patch: json_patch::Patch = serde_json::from_str(patch_json).unwrap();
        let (result, errors) = engine.apply_patches(&original, &patch);

        // First op should be skipped (null parent), second should succeed
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("parent path is null"));

        // The second patch should still have been applied
        assert_eq!(result["items"][0]["name"], "Updated Item");
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    #[serde(rename_all = "PascalCase")]
    enum ForecastModel {
        Auto,
        Mstl {
            #[serde(rename = "seasonalPeriods")]
            seasonal_periods: Vec<usize>,
        },
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct ModelWrapper {
        model: ForecastModel,
    }

    #[test]
    fn normalize_candidate_unflattens_external_enums() {
        let schema = ModelWrapper::gemini_schema();
        let mut candidate = json!({
            "model": {
                "type": "Mstl",
                "seasonalPeriods": [12]
            }
        });

        RefinementEngine::normalize_candidate_for_schema(&mut candidate, &schema);

        let parsed: ModelWrapper = serde_json::from_value(candidate).unwrap();
        assert_eq!(
            parsed,
            ModelWrapper {
                model: ForecastModel::Mstl {
                    seasonal_periods: vec![12]
                }
            }
        );
    }
}
