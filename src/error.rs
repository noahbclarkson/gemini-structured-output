use thiserror::Error;

/// Detailed error types for structured output operations.
#[derive(Debug, Error)]
pub enum StructuredError {
    #[error("Gemini client error: {0}")]
    Gemini(#[from] gemini_rust::ClientError),

    #[error("File API error: {0}")]
    Files(#[from] gemini_rust::FilesError),

    #[error("Cache API error: {0}")]
    Cache(#[from] gemini_rust::cache::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Patch error: {0}")]
    Patch(#[from] json_patch::PatchError),

    #[error("Invalid patch payload: {0}")]
    InvalidPatch(String),

    #[error("Schema generation failed: {0}")]
    Schema(String),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("Refinement exhausted after {retries} attempts. Last error: {last_error}")]
    RefinementExhausted { retries: usize, last_error: String },

    #[error("Context error: {0}")]
    Context(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {message}\n\nRaw response:\n{raw_text}\n\nSuggestion: {suggestion}")]
    ParseWithContext {
        message: String,
        raw_text: String,
        suggestion: String,
    },

    #[error("Tool execution failed: {tool_name} - {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Rate limited: retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },

    #[error("Service unavailable: {message}. Attempted {attempts} retries.")]
    ServiceUnavailable { message: String, attempts: usize },

    /// Workflow checkpoint triggered for human-in-the-loop processing.
    ///
    /// This error is intentionally raised by `CheckpointStep` to pause workflow
    /// execution and allow human review or modification of intermediate data.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match workflow.run(input, &ctx).await {
    ///     Ok(result) => println!("Completed: {:?}", result),
    ///     Err(StructuredError::Checkpoint { step_name, data }) => {
    ///         println!("Paused at '{}'. Data: {}", step_name, data);
    ///         // Save data for human review, then resume later
    ///     }
    ///     Err(e) => eprintln!("Error: {}", e),
    /// }
    /// ```
    #[error("Workflow checkpoint triggered at step '{step_name}'")]
    Checkpoint {
        /// The name of the checkpoint step that triggered the pause.
        step_name: String,
        /// Serialized intermediate data at the checkpoint.
        data: serde_json::Value,
    },
}

impl StructuredError {
    /// Create a parse error with helpful context.
    pub fn parse_error(err: serde_json::Error, raw_text: &str) -> Self {
        let suggestion = Self::suggest_parse_fix(&err, raw_text);
        Self::ParseWithContext {
            message: err.to_string(),
            raw_text: Self::truncate_for_display(raw_text, 500),
            suggestion,
        }
    }

    /// Create a tool execution error.
    pub fn tool_error(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolExecution {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } | Self::ServiceUnavailable { .. } => true,
            Self::Gemini(gemini_rust::ClientError::BadResponse { code, .. }) => {
                *code == 503 || *code == 429
            }
            _ => false,
        }
    }

    /// Check if this error is a workflow checkpoint.
    ///
    /// Checkpoints are intentional pauses for human-in-the-loop processing
    /// and should be handled differently from actual errors.
    pub fn is_checkpoint(&self) -> bool {
        matches!(self, Self::Checkpoint { .. })
    }

    /// Extract checkpoint data if this is a checkpoint error.
    ///
    /// Returns `Some((step_name, data))` if this is a checkpoint, `None` otherwise.
    pub fn checkpoint_data(&self) -> Option<(&str, &serde_json::Value)> {
        match self {
            Self::Checkpoint { step_name, data } => Some((step_name, data)),
            _ => None,
        }
    }

    /// Get suggested retry delay in seconds, if applicable.
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimited { retry_after_secs } => Some(*retry_after_secs),
            Self::ServiceUnavailable { .. } => Some(5),
            Self::Gemini(gemini_rust::ClientError::BadResponse {
                code: 429,
                description,
            }) => description
                .as_ref()
                .and_then(|d| parse_retry_delay_from_error(d)),
            _ => None,
        }
    }
}

/// Parse retry delay from Gemini API error response body.
fn parse_retry_delay_from_error(description: &str) -> Option<u64> {
    // 1. Try strict JSON parsing first (most reliable)
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(description) {
        if let Some(details) = json.get("error").and_then(|e| e.get("details")) {
            if let Some(arr) = details.as_array() {
                for detail in arr {
                    // Check for RetryInfo type
                    if detail.get("@type").and_then(|t| t.as_str())
                        == Some("type.googleapis.com/google.rpc.RetryInfo")
                    {
                        if let Some(delay_str) = detail.get("retryDelay").and_then(|d| d.as_str())
                        {
                            return parse_duration_string(delay_str);
                        }
                    }
                }
            }
        }
    }

    // 2. Fallback: Heuristic text search
    // Handles: "Please retry in 57s.", "retry in 488.04ms"
    let lower = description.to_lowercase();
    if let Some(idx) = lower.find("retry in ") {
        let start = idx + "retry in ".len();
        // extract the word after "retry in "
        let remainder = &lower[start..];
        let end = remainder
            .find(|c: char| !c.is_numeric() && c != '.' && c != 'm' && c != 's')
            .unwrap_or(remainder.len());

        let duration_str = &remainder[..end];
        return parse_duration_string(duration_str);
    }

    None
}

/// Parse duration strings like "44s", "44.5s", "500ms".
fn parse_duration_string(s: &str) -> Option<u64> {
    let s = s.trim();

    // Handle milliseconds
    if let Some(ms_part) = s.strip_suffix("ms") {
        if let Ok(ms) = ms_part.parse::<f64>() {
            // Convert to seconds, ensure at least 1 second if it's > 0 but < 1000ms
            // to be safe with rate limiters, or return 0 if 0.
            if ms <= 0.0 {
                return Some(0);
            }
            let secs = (ms / 1000.0).ceil() as u64;
            return Some(secs.max(1));
        }
    }

    // Handle seconds
    if let Some(s_part) = s.strip_suffix('s') {
        if let Ok(secs) = s_part.parse::<f64>() {
            return Some(secs.ceil() as u64);
        }
    }

    None
}

impl StructuredError {
    fn suggest_parse_fix(err: &serde_json::Error, raw_text: &str) -> String {
        let err_msg = err.to_string().to_lowercase();

        if err_msg.contains("expected a string") && err_msg.contains("map") {
            return "The model returned an object where a string was expected. \
                    Add #[schemars(description = \"...\")] to clarify the expected format."
                .to_string();
        }

        if err_msg.contains("expected value at line 1 column 1") {
            if raw_text.trim().is_empty() {
                return "The model returned an empty response. Try adding more context \
                        or adjusting the temperature."
                    .to_string();
            }
            if !raw_text.trim().starts_with(['{', '[']) {
                return "The model returned non-JSON text. This often happens when tools \
                        are enabled. The library will retry with strict JSON mode."
                    .to_string();
            }
        }

        if err_msg.contains("missing field") {
            return "The model omitted a required field. Consider making the field \
                    optional with Option<T> or adding a description."
                .to_string();
        }

        if err_msg.contains("invalid type") {
            return "Type mismatch in response. Check that your schema types match \
                    what the model is likely to return (e.g., use f64 for numbers)."
                .to_string();
        }

        "Check that your schema matches the expected response format. \
         Consider adding field descriptions with #[schemars(description = \"...\")]."
            .to_string()
    }

    fn truncate_for_display(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!(
                "{}... [truncated, {} total chars]",
                &text[..max_len],
                text.len()
            )
        }
    }
}

pub type Result<T> = std::result::Result<T, StructuredError>;

/// Extension trait for adding context to errors.
pub trait ResultExt<T> {
    /// Add context to an error.
    fn with_context(self, context: impl Into<String>) -> Result<T>;
}

impl<T, E: Into<StructuredError>> ResultExt<T> for std::result::Result<T, E> {
    fn with_context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| {
            let base_err = e.into();
            StructuredError::Context(format!("{}: {}", context.into(), base_err))
        })
    }
}
