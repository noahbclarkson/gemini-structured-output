//! Structured workflow execution events for tracing and observability.
//!
//! This module defines the event types that can be emitted during workflow execution,
//! enabling detailed tracking of step execution, intermediate artifacts, and errors.

use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

/// Events that can be emitted during workflow execution.
///
/// These events provide structured observability into workflow behavior,
/// replacing unstructured string logs with typed, serializable data.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "payload")]
pub enum WorkflowEvent {
    /// A step has started execution.
    StepStart {
        /// Name of the step being executed.
        step_name: String,
        /// The Rust type name of the input.
        input_type: String,
    },
    /// A step has finished successfully.
    StepEnd {
        /// Name of the step that completed.
        step_name: String,
        /// Duration of execution in milliseconds.
        duration_ms: u128,
    },
    /// An intermediate artifact was produced during execution.
    ///
    /// Useful for tracking outputs from individual steps in a chain,
    /// such as summaries, intermediate calculations, or partial results.
    Artifact {
        /// Name of the step that produced the artifact.
        step_name: String,
        /// Key identifying the artifact (e.g., "summary", "score").
        key: String,
        /// The artifact data as a JSON value.
        data: serde_json::Value,
    },
    /// An error occurred during step execution.
    Error {
        /// Name of the step where the error occurred.
        step_name: String,
        /// Error message describing what went wrong.
        message: String,
    },
}

/// A timestamped trace entry containing a workflow event.
///
/// Each trace entry records when the event occurred (as Unix epoch milliseconds)
/// along with the event itself.
#[derive(Debug, Clone, Serialize)]
pub struct TraceEntry {
    /// Unix epoch timestamp in milliseconds when this event occurred.
    pub timestamp: u128,
    /// The workflow event that was recorded.
    #[serde(flatten)]
    pub event: WorkflowEvent,
}

impl TraceEntry {
    /// Create a new trace entry with the current timestamp.
    pub fn new(event: WorkflowEvent) -> Self {
        let start = SystemTime::now();
        let timestamp = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis();
        Self { timestamp, event }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_entry_serialization() {
        let event = WorkflowEvent::StepStart {
            step_name: "Summarize".to_string(),
            input_type: "String".to_string(),
        };
        let entry = TraceEntry::new(event);

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"type\":\"StepStart\""));
        assert!(json.contains("\"step_name\":\"Summarize\""));
        assert!(json.contains("\"timestamp\":"));
    }

    #[test]
    fn test_artifact_event() {
        let event = WorkflowEvent::Artifact {
            step_name: "Analysis".to_string(),
            key: "score".to_string(),
            data: serde_json::json!({"value": 95, "confidence": 0.8}),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"Artifact\""));
        assert!(json.contains("\"key\":\"score\""));
    }
}
