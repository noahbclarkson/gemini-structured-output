//! Parallel step execution for concurrent workflow processing.
//!
//! This module provides `ParallelMapStep` which applies a worker step to multiple
//! inputs concurrently, with configurable concurrency limits.

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

        // We can share the context across parallel tasks because metrics are Arc<Mutex<...>>
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
