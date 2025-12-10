//! Windowed processing step for chunked parallel execution.
//!
//! This module provides `WindowedContextStep` which processes a list of items
//! in fixed-size windows with a shared context.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::Result;

use super::metrics::ExecutionContext;
use super::Step;

/// Process a list of items in fixed-size windows, running each window with shared context.
///
/// This step divides the input items into chunks of `window_size` and processes
/// each chunk concurrently (up to `concurrency` at a time) with the same context.
///
/// Input: `(Vec<Item>, Context)`
/// Output: `Vec<OutputItem>` (flattened results)
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{WindowedContextStep, ExecutionContext};
///
/// // Process 100 items in windows of 10, with 3 concurrent windows
/// let windowed = WindowedContextStep::new(batch_processor, 10, 3);
/// let ctx = ExecutionContext::new();
/// let results = windowed.run((items, shared_context), &ctx).await?;
/// ```
pub struct WindowedContextStep<Item, Context, OutputItem> {
    worker: Arc<dyn Step<(Vec<Item>, Context), Vec<OutputItem>>>,
    window_size: usize,
    concurrency: usize,
}

impl<Item, Context, OutputItem> WindowedContextStep<Item, Context, OutputItem>
where
    Item: Clone + Send + Sync + 'static,
    Context: Clone + Send + Sync + 'static,
    OutputItem: Send + Sync + 'static,
{
    /// Create a new windowed step with a worker, window size, and concurrency limit.
    pub fn new(
        worker: impl Step<(Vec<Item>, Context), Vec<OutputItem>> + 'static,
        window_size: usize,
        concurrency: usize,
    ) -> Self {
        Self {
            worker: Arc::new(worker),
            window_size: window_size.max(1),
            concurrency: concurrency.max(1),
        }
    }
}

#[async_trait]
impl<Item, Context, OutputItem> Step<(Vec<Item>, Context), Vec<OutputItem>>
    for WindowedContextStep<Item, Context, OutputItem>
where
    Item: Clone + Send + Sync + 'static,
    Context: Clone + Send + Sync + 'static,
    OutputItem: Send + Sync + 'static,
{
    async fn run(
        &self,
        (items, context): (Vec<Item>, Context),
        ctx: &ExecutionContext,
    ) -> Result<Vec<OutputItem>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let chunks: Vec<Vec<Item>> = items
            .chunks(self.window_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Share the execution context across parallel windows
        let results = stream::iter(chunks.into_iter().map(|chunk| {
            let worker = self.worker.clone();
            let user_context = context.clone();
            let exec_ctx = ctx.clone();
            async move { worker.run((chunk, user_context), &exec_ctx).await }
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        let mut outputs = Vec::new();
        for result in results {
            outputs.extend(result?);
        }

        Ok(outputs)
    }
}
