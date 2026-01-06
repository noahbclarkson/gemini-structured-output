use std::marker::PhantomData;
use std::path::Path;
use std::time::Duration;

use futures::{
    stream::{self, BoxStream},
    StreamExt,
};
use gemini_rust::{
    generation::model::UsageMetadata, Content, FileData, FileHandle, GenerationConfig, Message,
    Part, Role, SafetySetting, Tool,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, info, instrument, trace, warn};

use crate::{
    caching::CacheSettings,
    client::{BuilderOptions, MockRequest},
    error::StructuredError,
    models::GenerationOutcome,
    schema::{compile_validator, GeminiStructured},
    tools::ToolRegistry,
    Result, StructuredClient, StructuredValidator,
};

/// Fluent builder for structured requests targeting a specific output type.
pub struct StructuredRequest<'a, T> {
    client: &'a StructuredClient,
    contents: Vec<Content>,
    system_instruction: Option<String>,
    tools: Vec<Tool>,
    tool_registry: Option<ToolRegistry>,
    config: GenerationConfig,
    cache_settings: Option<CacheSettings>,
    safety_settings: Option<Vec<SafetySetting>>,
    refinement_instruction: Option<String>,
    max_tool_steps: usize,
    max_parse_attempts: usize,
    retry_count: usize,
    _marker: PhantomData<T>,
}

/// Streaming events emitted while a request is in-flight.
#[derive(Debug)]
pub enum StreamEvent<T> {
    /// A raw text chunk from the model (not yet parsed or validated).
    Chunk(String),
    /// Final structured output once streaming has completed.
    Complete(GenerationOutcome<T>),
}

impl<'a, T> StructuredRequest<'a, T>
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
    pub fn new(client: &'a StructuredClient) -> Self {
        Self {
            client,
            contents: Vec::new(),
            system_instruction: None,
            tools: Vec::new(),
            tool_registry: None,
            config: GenerationConfig::default(),
            cache_settings: None,
            safety_settings: None,
            refinement_instruction: None,
            max_tool_steps: 5,
            max_parse_attempts: 3,
            retry_count: 3,
            _marker: PhantomData,
        }
    }

    /// Set a system instruction.
    pub fn system(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = Some(instruction.into());
        self
    }

    /// Add a user text message.
    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.contents
            .push(Content::text(text).with_role(Role::User));
        self
    }

    /// Add a user message with an attached file handle (PDF/image).
    pub fn user_file(mut self, text: impl Into<String>, file: &FileHandle) -> Result<Self> {
        let meta = file.get_file_meta();

        let mut missing_fields = Vec::new();
        if meta.mime_type.is_none() {
            missing_fields.push("mime_type".to_string());
        }
        if meta.uri.is_none() {
            missing_fields.push("uri".to_string());
        }
        if !missing_fields.is_empty() {
            return Err(StructuredError::Context(format!(
                "incomplete file handle, missing {:?}",
                missing_fields
            )));
        }

        let mime_type = meta.mime_type.clone().unwrap();
        let file_uri = meta.uri.as_ref().unwrap().to_string();

        let content = Content {
            parts: Some(vec![
                Part::Text {
                    text: text.into(),
                    thought: None,
                    thought_signature: None,
                },
                Part::FileData {
                    file_data: FileData {
                        mime_type,
                        file_uri,
                    },
                },
            ]),
            role: Some(Role::User),
        };

        self.contents.push(content);
        Ok(self)
    }

    /// Upload a file from a local path and attach it as a user message.
    pub async fn add_file_path(self, path: impl AsRef<Path>) -> Result<Self> {
        let handle = self.client.file_manager.upload_path(path).await?;
        self.user_file("", &handle)
    }

    /// Add an explicit part list as a message.
    pub fn user_parts(mut self, parts: Vec<Part>) -> Self {
        let content = Content {
            parts: Some(parts),
            role: Some(Role::User),
        };
        self.contents.push(content);
        self
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Attach a ToolRegistry. If it has handlers, tool calls will be resolved automatically.
    pub fn with_tools(mut self, registry: ToolRegistry) -> Self {
        self.tools.extend(registry.definitions());
        self.tool_registry = Some(registry);
        self
    }

    /// Enable Google Search grounding.
    pub fn with_google_search(self) -> Self {
        self.with_tool(Tool::google_search())
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    /// Set top_p.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = Some(top_p);
        self
    }

    /// Enable Gemini thinking mode with a given budget.
    pub fn with_thinking(mut self, budget: i32, include_thoughts: bool) -> Self {
        self.config.thinking_config = Some(gemini_rust::ThinkingConfig {
            thinking_budget: Some(budget),
            include_thoughts: Some(include_thoughts),
            thinking_level: None,
        });
        self
    }

    /// Attach safety settings.
    pub fn with_safety_settings(mut self, safety: Vec<SafetySetting>) -> Self {
        self.safety_settings = Some(safety);
        self
    }

    /// Apply cache settings for this request.
    pub fn with_cache(mut self, settings: CacheSettings) -> Self {
        self.cache_settings = Some(settings);
        self
    }

    /// Override the generation config wholesale.
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Automatically refine the result using this instruction after generation.
    pub fn refine_with(mut self, instruction: impl Into<String>) -> Self {
        self.refinement_instruction = Some(instruction.into());
        self
    }

    /// Maximum tool-calling steps to prevent infinite loops.
    pub fn max_tool_steps(mut self, steps: usize) -> Self {
        self.max_tool_steps = steps.max(1);
        self
    }

    /// Maximum parse retries when the model returns invalid/empty JSON.
    pub fn max_parse_attempts(mut self, attempts: usize) -> Self {
        self.max_parse_attempts = attempts.max(1);
        self
    }

    /// Number of network retries for transient errors (503, 429).
    pub fn retries(mut self, count: usize) -> Self {
        self.retry_count = count;
        self
    }

    /// Execute the request and return parsed value plus metadata.
    #[instrument(skip_all, fields(target = std::any::type_name::<T>()))]
    pub async fn execute(mut self) -> Result<GenerationOutcome<T>> {
        if let Some(mock) = &self.client.mock_handler {
            let prompt_preview = self
                .contents
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join("\n---\n");
            let request = MockRequest {
                target: std::any::type_name::<T>().to_string(),
                system_instruction: self.system_instruction.clone(),
                prompt_preview,
            };
            let raw = (mock)(request)?;
            let parsed: T =
                serde_json::from_str(&raw).map_err(|e| StructuredError::parse_error(e, &raw))?;

            return Ok(GenerationOutcome::new(
                parsed,
                None,
                vec![],
                None,
                None,
                0,
                0,
            ));
        }

        let mut messages = Vec::new();
        for content in self.contents.drain(..) {
            let role = content.role.clone().unwrap_or(Role::User);
            messages.push(Message {
                role: role.clone(),
                content: content.with_role(role),
            });
        }

        let mut current_step = 0usize;
        let mut parse_attempts = 0usize;
        let mut total_network_attempts = 0usize;
        let mut escalated = false;

        loop {
            // Retry loop for 503/429 errors
            let mut response = None;
            let mut last_error = None;

            // If we are retrying due to a parsing error, we disable tools to force strict JSON mode.
            // This ensures the model conforms to the schema on the correction attempt.
            let tools_slice: &[Tool] = if parse_attempts > 0 {
                debug!("Disabling tools to force strict JSON mode for retry");
                &[]
            } else {
                &self.tools
            };

            // Determine which client to use based on escalation strategy
            let (active_client, is_escalated) = self.client.select_client(parse_attempts);
            if is_escalated && !escalated {
                info!(
                    parse_attempts = parse_attempts,
                    "Escalating to fallback model after parse failures"
                );
                escalated = true;
            }

            for attempt in 0..=self.retry_count {
                total_network_attempts += 1;

                let builder_result = self
                    .client
                    .configured_builder_with_client::<T>(
                        active_client,
                        &messages,
                        BuilderOptions {
                            tools: tools_slice,
                            config: &self.config,
                            cache_settings: &self.cache_settings,
                            system_instruction: &self.system_instruction,
                            safety_settings: &self.safety_settings,
                        },
                    )
                    .await;

                let builder = match builder_result {
                    Ok(b) => b,
                    Err(e) => {
                        last_error = Some(e);
                        break;
                    }
                };

                match builder.execute().await {
                    Ok(res) => {
                        response = Some(res);
                        break;
                    }
                    Err(e @ gemini_rust::ClientError::BadResponse { code, .. })
                        if code == 503 || code == 429 =>
                    {
                        let structured_err = StructuredError::Gemini(e);
                        // Use API-provided retry delay if available, otherwise exponential backoff
                        let delay_secs = structured_err
                            .retry_delay()
                            .unwrap_or_else(|| 2u64.pow(attempt as u32));
                        warn!(
                            "Attempt {}/{} failed with status {}. Retrying in {}s...",
                            attempt + 1,
                            self.retry_count + 1,
                            code,
                            delay_secs
                        );
                        last_error = Some(structured_err);
                        tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                    }
                    Err(e) => {
                        last_error = Some(StructuredError::Gemini(e));
                        break;
                    }
                }
            }

            let response = match response {
                Some(r) => r,
                None => {
                    return Err(last_error.unwrap_or_else(|| {
                        StructuredError::Context("Request failed after retries".to_string())
                    }))
                }
            };

            let function_calls: Vec<gemini_rust::tools::FunctionCall> =
                response.function_calls().into_iter().cloned().collect();

            if function_calls.is_empty() {
                let text = response.text();
                debug!(raw_response_length = text.len(), "Received model response");
                trace!(raw_response = %text, "Raw model text");

                if text.trim().is_empty() {
                    warn!("Received empty response from model");
                    messages.push(Message::user(
                        "The last response was empty. Return valid JSON matching the schema.",
                    ));
                    parse_attempts += 1;
                    if parse_attempts >= self.max_parse_attempts {
                        return Err(StructuredError::Context(
                            "Failed to get non-empty response".to_string(),
                        ));
                    }
                    continue;
                }

                // Clean the text to handle Markdown code blocks (e.g. ```json ... ```)
                let cleaned_text = clean_json_text(&text);
                if cleaned_text != text {
                    trace!(cleaned_response = %cleaned_text, "Cleaned JSON text");
                }

                // Parse to Value first, normalize maps (Array<__key__, __value__> -> Object), then deserialize to T
                match serde_json::from_str::<Value>(&cleaned_text) {
                    Ok(mut json_value) => {
                        // Apply normalization for HashMap schemas that were transformed to arrays
                        crate::schema::normalize_json_response(&mut json_value);

                        match serde_json::from_value::<T>(json_value) {
                            Ok(parsed) => {
                                debug!("Successfully parsed structured response");
                                if let Some(instruction) = &self.refinement_instruction {
                                    debug!("Starting refinement step");
                                    let refinement = self
                                        .client
                                        .refine(parsed, instruction.clone())
                                        .execute()
                                        .await?;
                                    return Ok(GenerationOutcome::new(
                                        refinement.value,
                                        response.usage_metadata,
                                        function_calls,
                                        response.model_version,
                                        response.response_id,
                                        parse_attempts,
                                        total_network_attempts,
                                    ));
                                }

                                return Ok(GenerationOutcome::new(
                                    parsed,
                                    response.usage_metadata,
                                    function_calls,
                                    response.model_version,
                                    response.response_id,
                                    parse_attempts,
                                    total_network_attempts,
                                ));
                            }
                            Err(err) => {
                                let validation_hint = validation_errors_for::<T>(&serde_json::from_str::<Value>(&cleaned_text).unwrap_or_default());
                                warn!(
                                    error = %err,
                                    raw_response = %text,
                                    cleaned_response = %cleaned_text,
                                    validation = ?validation_hint,
                                    "JSON parsing failed"
                                );
                                parse_attempts += 1;
                                if parse_attempts >= self.max_parse_attempts {
                                    let base = format!(
                                        "Failed to parse JSON after {} attempts: {err}",
                                        self.max_parse_attempts
                                    );
                                    if let Some(hint) = validation_hint {
                                        return Err(StructuredError::Validation(format!(
                                            "{base}; validation issues: {hint}; raw: {text}"
                                        )));
                                    }
                                    return Err(StructuredError::parse_error(err, &text));
                                }
                                let mut retry_msg = format!(
                                    "Failed to parse JSON: {err}. Return ONLY valid JSON matching the schema."
                                );
                                if let Some(hint) = validation_hint {
                                    retry_msg.push_str(&format!(" Validation issues: {hint}"));
                                }
                                messages.push(Message::user(retry_msg));
                                continue;
                            }
                        }
                    }
                    Err(err) => {
                        // JSON syntax error in the raw text itself
                        warn!(error = %err, raw_response = %text, "Failed to parse raw JSON syntax");
                        parse_attempts += 1;
                        if parse_attempts >= self.max_parse_attempts {
                            return Err(StructuredError::parse_error(err, &text));
                        }
                        messages.push(Message::user(format!(
                            "Failed to parse JSON: {err}. Return ONLY valid JSON matching the schema."
                        )));
                        continue;
                    }
                }
            }

            // Handle function calls (Tools)
            current_step += 1;
            if current_step > self.max_tool_steps {
                return Err(StructuredError::Context(
                    "Max tool steps exceeded".to_string(),
                ));
            }

            if let Some(candidate) = response.candidates.first() {
                messages.push(Message {
                    role: Role::Model,
                    content: candidate.content.clone(),
                });
            }

            let registry = self.tool_registry.as_ref().ok_or_else(|| {
                StructuredError::Context("Tool called but no registry provided".to_string())
            })?;

            debug!(count = function_calls.len(), "Processing tool calls");

            for call in function_calls {
                debug!(tool = %call.name, "Executing tool");
                let result_json = registry.execute(&call.name, call.args.clone()).await?;
                let content = gemini_rust::Content::function_response_json(&call.name, result_json)
                    .with_role(Role::User);
                messages.push(Message {
                    role: Role::User,
                    content,
                });
            }
        }
    }

    /// Stream raw text chunks before parsing into structured output.
    ///
    /// This is useful for UIs where you want to surface incremental model output
    /// while still validating against the target schema at the end.
    pub async fn stream(mut self) -> Result<BoxStream<'a, Result<StreamEvent<T>>>> {
        if let Some(mock) = &self.client.mock_handler {
            let prompt_preview = self
                .contents
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join("\n---\n");
            let request = MockRequest {
                target: std::any::type_name::<T>().to_string(),
                system_instruction: self.system_instruction.clone(),
                prompt_preview,
            };
            let raw = (mock)(request)?;
            let parsed: T =
                serde_json::from_str(&raw).map_err(|e| StructuredError::parse_error(e, &raw))?;
            let outcome = GenerationOutcome::new(parsed, None, vec![], None, None, 0, 0);
            return Ok(Box::pin(stream::once(async move {
                Ok(StreamEvent::Complete(outcome))
            })));
        }

        let mut messages = Vec::new();
        for content in self.contents.drain(..) {
            let role = content.role.clone().unwrap_or(Role::User);
            messages.push(Message {
                role: role.clone(),
                content: content.with_role(role),
            });
        }

        let builder = self
            .client
            .configured_builder::<T>(
                &messages,
                BuilderOptions {
                    tools: &self.tools,
                    config: &self.config,
                    cache_settings: &self.cache_settings,
                    system_instruction: &self.system_instruction,
                    safety_settings: &self.safety_settings,
                },
            )
            .await?;

        let inner_stream = builder.execute_stream().await?;

        struct StreamState<T> {
            inner: gemini_rust::GenerationStream,
            buffer: String,
            usage: Option<UsageMetadata>,
            model_version: Option<String>,
            response_id: Option<String>,
            function_calls: Vec<gemini_rust::tools::FunctionCall>,
            refinement_instruction: Option<String>,
            _marker: PhantomData<T>,
        }

        let state = StreamState::<T> {
            inner: inner_stream,
            buffer: String::new(),
            usage: None,
            model_version: None,
            response_id: None,
            function_calls: Vec::new(),
            refinement_instruction: self.refinement_instruction.clone(),
            _marker: PhantomData,
        };

        Ok(Box::pin(stream::try_unfold(
            state,
            move |mut state| async move {
                while let Some(resp) = state.inner.next().await {
                    let response = resp.map_err(StructuredError::Gemini)?;
                    if let Some(usage) = response.usage_metadata.clone() {
                        state.usage = Some(usage);
                    }
                    if let Some(version) = response.model_version.clone() {
                        state.model_version = Some(version);
                    }
                    if let Some(rid) = response.response_id.clone() {
                        state.response_id = Some(rid);
                    }

                    let calls: Vec<gemini_rust::tools::FunctionCall> =
                        response.function_calls().into_iter().cloned().collect();
                    if !calls.is_empty() {
                        state.function_calls.extend(calls);
                    }

                    let delta = response.text();
                    if !delta.is_empty() {
                        state.buffer.push_str(&delta);
                        return Ok(Some((StreamEvent::Chunk(delta), state)));
                    }
                }

                if state.buffer.is_empty() {
                    return Ok(None);
                }

                let cleaned = clean_json_text(&state.buffer);
                let mut json_value: Value = serde_json::from_str(&cleaned)
                    .map_err(|e| StructuredError::parse_error(e, &cleaned))?;
                crate::schema::normalize_json_response(&mut json_value);
                let parsed: T = serde_json::from_value(json_value)
                    .map_err(|e| StructuredError::parse_error(e, &cleaned))?;

                if let Some(instr) = &state.refinement_instruction {
                    return Err(StructuredError::Context(format!(
                        "refine_with(\"{instr}\") is not supported in streaming mode yet"
                    )));
                }

                let outcome = GenerationOutcome::new(
                    parsed,
                    state.usage.clone(),
                    state.function_calls.clone(),
                    state.model_version.clone(),
                    state.response_id.clone(),
                    0,
                    1,
                );

                state.buffer.clear();
                Ok(Some((StreamEvent::Complete(outcome), state)))
            },
        )))
    }
}

/// Helper to strip Markdown code blocks from the response text.
fn validation_errors_for<T: GeminiStructured>(value: &Value) -> Option<String> {
    let validator = compile_validator::<T>().ok()?;
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|err| format!("{}: {}", err.instance_path(), err))
        .collect();

    if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    }
}

/// Helper to strip Markdown code blocks from the response text.
pub(crate) fn clean_json_text(text: &str) -> String {
    let text = text.trim();

    // Check for standard markdown code blocks
    if let Some(start) = text.find("```") {
        if let Some(end) = text.rfind("```") {
            if start < end {
                // Find the newline after the first ``` (skipping "json" or "xml" etc)
                if let Some(newline) = text[start..end].find('\n') {
                    let content_start = start + newline + 1;
                    if content_start < end {
                        return text[content_start..end].trim().to_string();
                    }
                }
            }
        }
    }

    // Fallback heuristic: find first '{' or '[' and last '}' or ']'
    if let Some(start) = text.find(['{', '[']) {
        if let Some(end) = text.rfind(['}', ']']) {
            if start <= end {
                return text[start..=end].to_string();
            }
        }
    }
    // Return as is if no heuristics matched
    text.to_string()
}
