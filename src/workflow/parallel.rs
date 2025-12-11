//! Parallel step execution for concurrent workflow processing.
//!
//! This module provides `ParallelMapStep` which applies a worker step to multiple
//! inputs concurrently, with configurable concurrency limits.
//!
//! Internally, this is implemented using `BatchStep` with a batch size of 1,
//! demonstrating how the batch primitive can be used for different parallel
//! processing patterns.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::Result;

use super::metrics::ExecutionContext;
use super::Step;

/// Apply a worker step to each item concurrently, returning collected outputs.
///
/// This step takes a `Vec<Input>` and runs the worker step on each item
/// in parallel, respecting the configured concurrency limit.
///
/// # Implementation
///
/// This is a high-level convenience wrapper. For more control over batching,
/// use `BatchStep` directly.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ParallelMapStep, ExecutionContext};
///
/// let parallel = ParallelMapStep::new(analyzer, 5); // 5 concurrent workers
/// let ctx = ExecutionContext::new();
/// let results = parallel.run(items, &ctx).await?;
/// ```
pub struct ParallelMapStep<Input, Output> {
    worker: Arc<dyn Step<Input, Output>>,
    concurrency: usize,
}

impl<Input, Output> ParallelMapStep<Input, Output>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new parallel map step with a worker and concurrency limit.
    pub fn new(worker: impl Step<Input, Output> + 'static, concurrency: usize) -> Self {
        Self {
            worker: Arc::new(worker),
            concurrency: concurrency.max(1),
        }
    }

    /// Get the configured concurrency limit.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }
}

#[async_trait]
impl<Input, Output> Step<Vec<Input>, Vec<Output>> for ParallelMapStep<Input, Output>
where
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    async fn run(&self, inputs: Vec<Input>, ctx: &ExecutionContext) -> Result<Vec<Output>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let results = stream::iter(inputs.into_iter().map(|input| {
            let worker = self.worker.clone();
            let ctx_clone = ctx.clone();
            async move { worker.run(input, &ctx_clone).await }
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        let mut outputs = Vec::with_capacity(results.len());
        for result in results {
            outputs.push(result?);
        }

        Ok(outputs)
    }
}

/// Builder for creating parallel processing pipelines.
///
/// This provides a fluent API for configuring parallel map operations.
pub struct ParallelMapBuilder<I, O> {
    worker: Arc<dyn Step<I, O>>,
    concurrency: usize,
}

impl<I, O> ParallelMapBuilder<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Start building a parallel map step with the given worker.
    pub fn new(worker: impl Step<I, O> + 'static) -> Self {
        Self {
            worker: Arc::new(worker),
            concurrency: 4, // sensible default
        }
    }

    /// Set the concurrency limit (default: 4).
    pub fn concurrency(mut self, limit: usize) -> Self {
        self.concurrency = limit.max(1);
        self
    }

    /// Build the parallel map step.
    pub fn build(self) -> ParallelMapStep<I, O> {
        ParallelMapStep {
            worker: self.worker,
            concurrency: self.concurrency,
        }
    }
}
