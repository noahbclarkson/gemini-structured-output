//! Pass-through step for side-effect inspection during workflow execution.
//!
//! The `TapStep` combinator allows users to inspect intermediate data in fluent
//! chains without changing the return types or breaking the chain structure.

use std::sync::Arc;

use async_trait::async_trait;

use crate::Result;

use super::metrics::ExecutionContext;
use super::Step;

/// A pass-through step that executes a side-effect function.
///
/// `TapStep` wraps an inner step and runs a user-provided function on the output
/// before passing it through unchanged. This is useful for logging, debugging,
/// or emitting custom artifacts during workflow execution.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{Step, ExecutionContext};
///
/// let pipeline = summarizer
///     .tap(|summary, ctx| {
///         println!("Summary length: {}", summary.text.len());
///         ctx.emit_artifact("Summarize", "word_count", &summary.word_count);
///     })
///     .then(email_drafter);
///
/// let ctx = ExecutionContext::new();
/// let result = pipeline.run(input, &ctx).await?;
/// ```
pub struct TapStep<S, F, I, O> {
    inner: S,
    func: Arc<F>,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<S, F, I, O> TapStep<S, F, I, O> {
    /// Create a new tap step wrapping an inner step with an inspection function.
    pub fn new(inner: S, func: F) -> Self {
        Self {
            inner,
            func: Arc::new(func),
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S, F, I, O> Step<I, O> for TapStep<S, F, I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    S: Step<I, O> + Send + Sync,
    F: Fn(&O, &ExecutionContext) + Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let output = self.inner.run(input, ctx).await?;
        (self.func)(&output, ctx);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DoubleStep;

    #[async_trait]
    impl Step<i32, i32> for DoubleStep {
        async fn run(&self, input: i32, _ctx: &ExecutionContext) -> Result<i32> {
            Ok(input * 2)
        }
    }

    #[tokio::test]
    async fn test_tap_executes_side_effect() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let step = TapStep::new(DoubleStep, move |_output: &i32, _ctx: &ExecutionContext| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let ctx = ExecutionContext::new();
        let result = step.run(5, &ctx).await.unwrap();

        assert_eq!(result, 10);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_tap_passes_through_unchanged() {
        let step = TapStep::new(DoubleStep, |output: &i32, _ctx: &ExecutionContext| {
            assert_eq!(*output, 10);
        });

        let ctx = ExecutionContext::new();
        let result = step.run(5, &ctx).await.unwrap();

        assert_eq!(result, 10);
    }
}
