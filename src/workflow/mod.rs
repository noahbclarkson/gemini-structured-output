//! Workflow orchestration primitives for composing multi-step pipelines.
//!
//! This module provides building blocks for creating complex workflows
//! with type-safe step chaining, parallel execution, aggregation, and
//! observability through execution metrics.
//!
//! # Core Concepts
//!
//! - **Step**: The fundamental trait for workflow units
//! - **ExecutionContext**: Shared context for metrics collection
//! - **WorkflowMetrics**: Aggregated token usage and execution statistics
//! - **ChainStep**: Sequential composition of steps
//! - **ChainTupleStep**: Sequential composition preserving intermediate results
//! - **MapStep**: Inline transformations between steps
//! - **ParallelMapStep**: Apply a step to multiple inputs concurrently
//! - **ReduceStep**: Aggregate multiple results into one
//! - **RouterStep**: Conditional branching based on LLM decisions
//! - **Workflow**: High-level container with automatic metrics collection
//!
//! # Example: Fluent Pipeline with Metrics
//!
//! ```rust,ignore
//! use gemini_structured_output::workflow::{Step, Workflow, ExecutionContext};
//!
//! // Chain steps fluently
//! let pipeline = summarizer
//!     .map(|summary| (summary.clone(), calculate_score(&summary)))
//!     .then(reviewer);
//!
//! // Run with automatic metrics collection
//! let workflow = Workflow::new(pipeline).with_name("SummaryReview");
//! let (result, metrics) = workflow.run(input).await?;
//!
//! println!("Total tokens: {}", metrics.total_token_count);
//! ```

mod batch;
mod chain;
mod checkpoint;
mod events;
mod instrumented;
mod legacy;
mod metrics;
mod parallel;
mod reduce;
mod review;
mod router;
mod state;
mod tap;
mod traits;
mod windowed;

pub use batch::{BatchStep, SingleItemAdapter};
pub use chain::{ChainStep, ChainTupleStep};
pub use checkpoint::{CheckpointStep, ConditionalCheckpointStep};
pub use events::{TraceEntry, WorkflowEvent};
pub use instrumented::InstrumentedStep;
pub use legacy::{WorkflowAction, WorkflowFuture, WorkflowStep};
pub use metrics::{ExecutionContext, WorkflowMetrics};
pub use parallel::{ParallelMapBuilder, ParallelMapStep};
pub use reduce::{ConfiguredReduceStep, ReduceStep, ReduceStepBuilder};
pub use review::ReviewStep;
pub use router::RouterStep;
pub use state::{LambdaStateStep, StateStep, StateWorkflow, StepAdapter};
pub use tap::TapStep;
pub use traits::{BoxedStepExt, LambdaStep, MapStep, Step};
pub use windowed::WindowedContextStep;

use std::sync::Arc;

use crate::Result;

/// A high-level container for a workflow process with automatic metrics collection.
///
/// `Workflow` wraps a step (or chain of steps) and provides:
/// - Automatic `ExecutionContext` creation and management
/// - Aggregated metrics from all steps
/// - Optional workflow naming for logging
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{Workflow, Step};
///
/// let pipeline = step_a.then(step_b).then(step_c);
/// let workflow = Workflow::new(pipeline).with_name("MyPipeline");
///
/// let (result, metrics) = workflow.run(input).await?;
/// println!("Completed {} steps using {} tokens",
///     metrics.steps_completed,
///     metrics.total_token_count
/// );
/// ```
pub struct Workflow<Input, Output> {
    step: Arc<dyn Step<Input, Output>>,
    name: Option<String>,
}

impl<Input, Output> Workflow<Input, Output>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new workflow wrapping the given step.
    pub fn new(step: impl Step<Input, Output> + 'static) -> Self {
        Self {
            step: Arc::new(step),
            name: None,
        }
    }

    /// Set a name for this workflow (used in logging).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Run the workflow and return the result along with execution metrics.
    ///
    /// This method:
    /// 1. Creates a fresh `ExecutionContext`
    /// 2. Passes it through all steps in the pipeline
    /// 3. Collects and returns aggregated metrics
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (result, metrics) = workflow.run(input).await?;
    ///
    /// if !metrics.failures.is_empty() {
    ///     eprintln!("Workflow had {} failures", metrics.failures.len());
    /// }
    /// ```
    pub async fn run(&self, input: Input) -> Result<(Output, WorkflowMetrics)> {
        let ctx = ExecutionContext::new();

        if let Some(name) = &self.name {
            tracing::info!("Starting workflow: {}", name);
        }

        let result = self.step.run(input, &ctx).await;
        if let Err(err) = &result {
            ctx.record_failure(err.to_string());
        }

        let metrics = ctx.snapshot();

        if let Some(name) = &self.name {
            match &result {
                Ok(_) => {
                    tracing::info!(
                        "Workflow '{}' completed. Steps: {}, Tokens: {} (Prompt: {}, Completion: {})",
                        name,
                        metrics.steps_completed,
                        metrics.total_token_count,
                        metrics.prompt_token_count,
                        metrics.candidates_token_count
                    );
                }
                Err(e) => {
                    tracing::error!("Workflow '{}' failed: {}", name, e);
                }
            }
        }

        result.map(|output| (output, metrics))
    }

    /// Run the workflow with a pre-existing execution context.
    ///
    /// This is useful when you want to share metrics across multiple workflows
    /// or when integrating into a larger execution context.
    pub async fn run_with_context(&self, input: Input, ctx: &ExecutionContext) -> Result<Output> {
        if let Some(name) = &self.name {
            tracing::info!("Starting workflow: {}", name);
        }

        let result = self.step.run(input, ctx).await;
        if let Err(err) = &result {
            ctx.record_failure(err.to_string());
        }

        if let Some(name) = &self.name {
            match &result {
                Ok(_) => tracing::info!("Workflow '{}' completed", name),
                Err(e) => tracing::error!("Workflow '{}' failed: {}", name, e),
            }
        }

        result
    }
}
