//! Metrics and context for workflow execution.
//!
//! This module provides observability primitives for tracking workflow execution,
//! including token usage, retry attempts, failure logging, and structured event tracing.

use std::sync::{Arc, Mutex};

use gemini_rust::generation::model::UsageMetadata;
use serde::Serialize;

use super::events::{TraceEntry, WorkflowEvent};
use crate::models::GenerationOutcome;

/// Aggregated metrics for a workflow execution.
#[derive(Debug, Default, Clone)]
pub struct WorkflowMetrics {
    /// Total prompt tokens consumed across all steps.
    pub prompt_token_count: usize,
    /// Total response tokens generated across all steps.
    pub candidates_token_count: usize,
    /// Total tokens (prompt + response) across all steps.
    pub total_token_count: usize,
    /// Total network attempts (including retries) across all steps.
    pub network_attempts: usize,
    /// Total parse correction attempts across all steps.
    pub parse_attempts: usize,
    /// Number of workflow steps completed successfully.
    pub steps_completed: usize,
    /// Collected failure messages from the workflow.
    pub failures: Vec<String>,
}

impl WorkflowMetrics {
    /// Add usage metadata from a generation response.
    pub fn add_usage(&mut self, usage: &Option<UsageMetadata>) {
        if let Some(u) = usage {
            self.prompt_token_count += u.prompt_token_count.unwrap_or(0) as usize;
            self.candidates_token_count += u.candidates_token_count.unwrap_or(0) as usize;
            self.total_token_count += u.total_token_count.unwrap_or(0) as usize;
        }
    }

    /// Record network and parse attempt counts.
    pub fn record_attempts(&mut self, network: usize, parse: usize) {
        self.network_attempts += network;
        self.parse_attempts += parse;
    }

    /// Record a failure message.
    pub fn record_failure(&mut self, error: String) {
        self.failures.push(error);
    }

    /// Increment the steps completed counter.
    pub fn record_step(&mut self) {
        self.steps_completed += 1;
    }
}

/// Context passed to every step in the workflow.
///
/// This context is cloneable and thread-safe, allowing it to be shared
/// across parallel step executions. All metric updates are synchronized.
///
/// # Tracing
///
/// The context also maintains a structured trace log of workflow events,
/// enabling detailed observability without relying on unstructured string logs.
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ExecutionContext, WorkflowEvent};
///
/// let ctx = ExecutionContext::new();
/// ctx.emit(WorkflowEvent::StepStart {
///     step_name: "Summarize".to_string(),
///     input_type: "String".to_string(),
/// });
///
/// // Later, get all trace entries
/// let traces = ctx.trace_snapshot();
/// for entry in traces {
///     println!("{:?}", entry);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Shared metrics accumulator.
    pub metrics: Arc<Mutex<WorkflowMetrics>>,
    /// Shared trace log for structured workflow events.
    pub traces: Arc<Mutex<Vec<TraceEntry>>>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new execution context with empty metrics and traces.
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(WorkflowMetrics::default())),
            traces: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record usage and attempt counts from a generation outcome.
    pub fn record_outcome<T>(&self, outcome: &GenerationOutcome<T>) {
        let mut m = self.metrics.lock().unwrap();
        m.add_usage(&outcome.usage);
        m.record_attempts(outcome.network_attempts, outcome.parse_attempts);
    }

    /// Increment the steps completed counter.
    pub fn record_step(&self) {
        let mut m = self.metrics.lock().unwrap();
        m.record_step();
    }

    /// Record a failure message.
    pub fn record_failure(&self, error: impl Into<String>) {
        let mut m = self.metrics.lock().unwrap();
        m.record_failure(error.into());
    }

    /// Get a snapshot of the current metrics.
    pub fn snapshot(&self) -> WorkflowMetrics {
        let m = self.metrics.lock().unwrap();
        m.clone()
    }

    /// Emit a structured workflow event to the trace log.
    ///
    /// Events are timestamped automatically when emitted.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ctx.emit(WorkflowEvent::StepStart {
    ///     step_name: "Summarize".to_string(),
    ///     input_type: "Article".to_string(),
    /// });
    /// ```
    pub fn emit(&self, event: WorkflowEvent) {
        let entry = TraceEntry::new(event);
        self.traces.lock().unwrap().push(entry);
    }

    /// Emit an artifact event with automatic JSON serialization.
    ///
    /// This is a convenience method for recording intermediate outputs
    /// from workflow steps.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let summary = Summary { text: "...".to_string(), word_count: 150 };
    /// ctx.emit_artifact("Summarize", "output", &summary);
    /// ```
    pub fn emit_artifact<T: Serialize>(&self, step_name: &str, key: &str, data: &T) {
        let json_data = serde_json::to_value(data)
            .unwrap_or_else(|_| serde_json::json!("<serialization_error>"));
        self.emit(WorkflowEvent::Artifact {
            step_name: step_name.to_string(),
            key: key.to_string(),
            data: json_data,
        });
    }

    /// Get a snapshot of the current trace log.
    ///
    /// Returns all trace entries recorded so far. Useful for debugging
    /// or exporting execution traces.
    pub fn trace_snapshot(&self) -> Vec<TraceEntry> {
        self.traces.lock().unwrap().clone()
    }

    /// Clear all trace entries.
    ///
    /// This can be useful when reusing a context across multiple workflow runs.
    pub fn clear_traces(&self) {
        self.traces.lock().unwrap().clear();
    }
}
