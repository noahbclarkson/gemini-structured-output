//! Batch processing step for chunked parallel execution.
//!
//! This module provides `BatchStep` which processes items in configurable batches
//! with a shared context, reducing code duplication between parallel processing
//! implementations.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::Result;

use super::metrics::ExecutionContext;
use super::Step;

/// Processes items in batches with a shared context.
///
/// `BatchStep` divides input items into chunks of `batch_size` and processes
/// each chunk concurrently (up to `concurrency` at a time) with a shared context.
///
/// This is a low-level primitive that powers both `ParallelMapStep` and
/// `WindowedContextStep`, reducing code duplication while providing flexibility.
///
/// # Type Parameters
///
/// * `Item` - The type of individual items to process
/// * `Context` - The type of shared context passed to each batch
/// * `OutputItem` - The type of items produced by processing
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{BatchStep, ExecutionContext, Step};
///
/// // Create a step that processes batches of documents with shared config
/// let batch_processor = BatchStep::new(
///     document_analyzer,  // Step<(Vec<Document>, Config), Vec<Analysis>>
///     10,                 // batch_size
///     3,                  // concurrency
/// );
///
/// let ctx = ExecutionContext::new();
/// let results = batch_processor.run((documents, config), &ctx).await?;
/// ```
pub struct BatchStep<Item, Context, OutputItem> {
    worker: Arc<dyn Step<(Vec<Item>, Context), Vec<OutputItem>>>,
    batch_size: usize,
    concurrency: usize,
}

impl<Item, Context, OutputItem> BatchStep<Item, Context, OutputItem>
where
    Item: Clone + Send + Sync + 'static,
    Context: Clone + Send + Sync + 'static,
    OutputItem: Send + Sync + 'static,
{
    /// Create a new batch step with a worker, batch size, and concurrency limit.
    ///
    /// # Arguments
    ///
    /// * `worker` - The step that processes each batch
    /// * `batch_size` - Number of items per batch (minimum 1)
    /// * `concurrency` - Maximum number of concurrent batch operations (minimum 1)
    pub fn new(
        worker: impl Step<(Vec<Item>, Context), Vec<OutputItem>> + 'static,
        batch_size: usize,
        concurrency: usize,
    ) -> Self {
        Self {
            worker: Arc::new(worker),
            batch_size: batch_size.max(1),
            concurrency: concurrency.max(1),
        }
    }

    /// Get the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the configured concurrency limit.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }
}

#[async_trait]
impl<Item, Context, OutputItem> Step<(Vec<Item>, Context), Vec<OutputItem>>
    for BatchStep<Item, Context, OutputItem>
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
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

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

/// Adapter that converts a `Step<I, O>` into `Step<(Vec<I>, ()), Vec<O>>`.
///
/// This allows single-item steps to be used with `BatchStep` by processing
/// each item in the batch individually.
pub struct SingleItemAdapter<I, O> {
    inner: Arc<dyn Step<I, O>>,
}

impl<I, O> SingleItemAdapter<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Create a new adapter wrapping a single-item step.
    pub fn new(inner: impl Step<I, O> + 'static) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[async_trait]
impl<I, O> Step<(Vec<I>, ()), Vec<O>> for SingleItemAdapter<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn run(&self, (inputs, _): (Vec<I>, ()), ctx: &ExecutionContext) -> Result<Vec<O>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for item in inputs {
            outputs.push(self.inner.run(item, ctx).await?);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::LambdaStep;

    #[tokio::test]
    async fn test_batch_step_basic() {
        let worker = LambdaStep(|(items, multiplier): (Vec<i32>, i32)| async move {
            Ok(items.into_iter().map(|x| x * multiplier).collect::<Vec<_>>())
        });

        let batch = BatchStep::new(worker, 2, 2);
        let ctx = ExecutionContext::new();

        let items = vec![1, 2, 3, 4, 5];
        let result = batch.run((items, 10), &ctx).await.unwrap();

        assert_eq!(result.len(), 5);
        assert!(result.contains(&10));
        assert!(result.contains(&20));
        assert!(result.contains(&30));
        assert!(result.contains(&40));
        assert!(result.contains(&50));
    }

    #[tokio::test]
    async fn test_batch_step_empty_input() {
        let worker = LambdaStep(|(items, _): (Vec<i32>, ())| async move { Ok(items) });

        let batch = BatchStep::new(worker, 2, 2);
        let ctx = ExecutionContext::new();

        let result = batch.run((vec![], ()), &ctx).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_single_item_adapter() {
        let doubler = LambdaStep(|x: i32| async move { Ok(x * 2) });
        let adapter = SingleItemAdapter::new(doubler);

        let ctx = ExecutionContext::new();
        let result = adapter.run((vec![1, 2, 3], ()), &ctx).await.unwrap();

        assert_eq!(result, vec![2, 4, 6]);
    }
}
