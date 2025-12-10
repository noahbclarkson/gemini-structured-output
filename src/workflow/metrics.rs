//! Metrics and context for workflow execution.
//!
//! This module provides observability primitives for tracking workflow execution,
//! including token usage, retry attempts, and failure logging.

use std::sync::{Arc, Mutex};

use gemini_rust::generation::model::UsageMetadata;

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
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Shared metrics accumulator.
    pub metrics: Arc<Mutex<WorkflowMetrics>>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new execution context with empty metrics.
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(WorkflowMetrics::default())),
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
}
