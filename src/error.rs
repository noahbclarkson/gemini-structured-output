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

    /// Get suggested retry delay in seconds, if applicable.
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimited { retry_after_secs } => Some(*retry_after_secs),
            Self::ServiceUnavailable { .. } => Some(5),
            _ => None,
        }
    }

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
