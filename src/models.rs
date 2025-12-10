/// Information about an individual refinement attempt.
#[derive(Debug, Clone)]
pub struct RefinementAttempt {
    pub patch: String,
    pub success: bool,
    pub error: Option<String>,
}

impl RefinementAttempt {
    pub fn success(patch: String) -> Self {
        Self {
            patch,
            success: true,
            error: None,
        }
    }

    pub fn failure(patch: String, error: impl Into<String>) -> Self {
        Self {
            patch,
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Outcome of the refinement loop including the final value and patch trace.
#[derive(Debug, Clone)]
pub struct RefinementOutcome<T> {
    pub value: T,
    pub attempts: Vec<RefinementAttempt>,
}

impl<T> RefinementOutcome<T> {
    pub fn new(value: T, attempts: Vec<RefinementAttempt>) -> Self {
        Self { value, attempts }
    }
}

/// Structured generation result with additional metadata.
#[derive(Debug, Clone)]
pub struct GenerationOutcome<T> {
    pub value: T,
    pub usage: Option<gemini_rust::generation::model::UsageMetadata>,
    pub function_calls: Vec<gemini_rust::tools::FunctionCall>,
    pub model_version: Option<String>,
    pub response_id: Option<String>,
    /// How many parse correction attempts were needed.
    pub parse_attempts: usize,
    /// How many network calls (including retries) were made.
    pub network_attempts: usize,
}

impl<T> GenerationOutcome<T> {
    pub fn new(
        value: T,
        usage: Option<gemini_rust::generation::model::UsageMetadata>,
        function_calls: Vec<gemini_rust::tools::FunctionCall>,
        model_version: Option<String>,
        response_id: Option<String>,
        parse_attempts: usize,
        network_attempts: usize,
    ) -> Self {
        Self {
            value,
            usage,
            function_calls,
            model_version,
            response_id,
            parse_attempts,
            network_attempts,
        }
    }
}
