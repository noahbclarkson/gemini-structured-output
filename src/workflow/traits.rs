//! Core workflow traits and primitives.

use std::sync::Arc;

use async_trait::async_trait;

use crate::Result;

use super::chain::{ChainStep, ChainTupleStep};
use super::metrics::ExecutionContext;

/// A unit of asynchronous work that transforms an input into an output.
///
/// This is the fundamental building block for composing workflows.
/// Steps can be chained together using the `.then()` method.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::Step;
///
/// // Create a pipeline by chaining steps
/// let pipeline = step_a.then(step_b).then(step_c);
/// let ctx = ExecutionContext::new();
/// let result = pipeline.run(input, &ctx).await?;
/// ```
#[async_trait]
pub trait Step<Input, Output>: Send + Sync {
    /// Execute this step with the given input and execution context.
    async fn run(&self, input: Input, ctx: &ExecutionContext) -> Result<Output>;

    /// Chain this step with another step, creating a pipeline.
    ///
    /// The output of this step becomes the input of the next step.
    /// The compiler guarantees type safety between steps.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // RawText -> Summary -> EmailDraft
    /// let pipeline = summarizer.then(email_drafter);
    /// let email = pipeline.run(raw_text, &ctx).await?;
    /// ```
    fn then<NextOut, S>(self, next: S) -> ChainStep<Input, Output, NextOut>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Send + Sync + 'static,
        NextOut: Send + Sync + 'static,
        S: Step<Output, NextOut> + 'static,
    {
        ChainStep::new(self, next)
    }

    /// Chain with another step, returning both intermediate and final results.
    ///
    /// This is useful when you need to pass the intermediate result along
    /// with the final result for further processing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // RawText -> (Summary, EmailDraft) - keep both!
    /// let pipeline = summarizer.then_tuple(email_drafter);
    /// let (summary, email) = pipeline.run(raw_text, &ctx).await?;
    /// ```
    fn then_tuple<NextOut, S>(self, next: S) -> ChainTupleStep<Input, Output, NextOut>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Clone + Send + Sync + 'static,
        NextOut: Send + Sync + 'static,
        S: Step<Output, NextOut> + 'static,
    {
        ChainTupleStep::new(self, next)
    }

    /// Transform the output of this step using a function.
    ///
    /// This is useful for calculations, formatting, or enriching data (e.g., creating tuples)
    /// before passing it to the next step.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// step.map(|data| {
    ///     let derived = calculate_score(&data);
    ///     (data, derived) // Pass tuple to next step
    /// })
    /// ```
    fn map<NewOut, F>(self, f: F) -> MapStep<Self, F, Input, Output, NewOut>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Send + Sync + 'static,
        NewOut: Send + Sync + 'static,
        F: Fn(Output) -> NewOut + Send + Sync + 'static,
    {
        MapStep::new(self, f)
    }

    /// Inspect the output of this step without modifying it.
    ///
    /// This is useful for logging, debugging, or emitting custom artifacts
    /// during workflow execution. The function receives both the output and
    /// the execution context, allowing it to record traces or metrics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = summarizer
    ///     .tap(|summary, ctx| {
    ///         println!("Generated summary: {}", summary.text);
    ///         ctx.emit_artifact("Summarize", "length", &summary.text.len());
    ///     })
    ///     .then(email_drafter);
    /// ```
    fn tap<F>(self, func: F) -> super::tap::TapStep<Self, F, Input, Output>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Send + Sync + 'static,
        F: Fn(&Output, &ExecutionContext) + Send + Sync + 'static,
    {
        super::tap::TapStep::new(self, func)
    }

    /// Wrap this step with automatic start/end event instrumentation.
    ///
    /// When the step runs, it will automatically emit:
    /// - `StepStart` event when execution begins
    /// - `StepEnd` event on success (with duration)
    /// - `Error` event on failure
    ///
    /// This provides comprehensive workflow tracing without manual instrumentation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Chain instrumented steps for full execution tracing
    /// let pipeline = summarizer.named("Summarize")
    ///     .then(drafter.named("DraftEmail"))
    ///     .then(sender.named("SendEmail"));
    ///
    /// let ctx = ExecutionContext::new();
    /// let result = pipeline.run(input, &ctx).await?;
    ///
    /// // View the execution timeline
    /// for trace in ctx.trace_snapshot() {
    ///     println!("{:?}", trace);
    /// }
    /// ```
    fn named(self, name: impl Into<String>) -> super::instrumented::InstrumentedStep<Self>
    where
        Self: Sized,
    {
        super::instrumented::InstrumentedStep::new(self, name)
    }
}

/// Convenience wrapper to turn an async function or closure into a [`Step`].
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::LambdaStep;
///
/// let double = LambdaStep(|x: i32| async move { Ok(x * 2) });
/// let ctx = ExecutionContext::new();
/// let result = double.run(5, &ctx).await?; // Returns 10
/// ```
pub struct LambdaStep<F>(pub F);

#[async_trait]
impl<F, Fut, Input, Output> Step<Input, Output> for LambdaStep<F>
where
    Input: Send + 'static,
    Output: Send + 'static,
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Output>> + Send,
{
    async fn run(&self, input: Input, _ctx: &ExecutionContext) -> Result<Output> {
        (self.0)(input).await
    }
}

/// Step that applies a transformation function to the output of a previous step.
///
/// Created by calling `.map()` on any `Step`.
pub struct MapStep<S, F, I, O, NewO> {
    inner: S,
    func: Arc<F>,
    _marker: std::marker::PhantomData<(I, O, NewO)>,
}

impl<S, F, I, O, NewO> MapStep<S, F, I, O, NewO> {
    /// Create a new map step wrapping an inner step with a transformation function.
    pub fn new(inner: S, func: F) -> Self {
        Self {
            inner,
            func: Arc::new(func),
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S, F, I, O, NewO> Step<I, NewO> for MapStep<S, F, I, O, NewO>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    NewO: Send + Sync + 'static,
    S: Step<I, O> + Send + Sync,
    F: Fn(O) -> NewO + Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<NewO> {
        let output = self.inner.run(input, ctx).await?;
        Ok((self.func)(output))
    }
}

/// Extension trait for boxed steps to enable chaining.
///
/// This is useful when working with trait objects that have been boxed.
pub trait BoxedStepExt<Input, Output>: Step<Input, Output> {
    /// Chain this boxed step with another step.
    fn then_boxed<NextOut, S>(
        self: Box<Self>,
        next: S,
    ) -> Box<dyn Step<Input, NextOut> + Send + Sync>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Send + Sync + 'static,
        NextOut: Send + Sync + 'static,
        S: Step<Output, NextOut> + 'static;
}

impl<T, Input, Output> BoxedStepExt<Input, Output> for T
where
    T: Step<Input, Output> + ?Sized,
{
    fn then_boxed<NextOut, S>(
        self: Box<Self>,
        next: S,
    ) -> Box<dyn Step<Input, NextOut> + Send + Sync>
    where
        Self: Sized + 'static,
        Input: Send + Sync + 'static,
        Output: Send + Sync + 'static,
        NextOut: Send + Sync + 'static,
        S: Step<Output, NextOut> + 'static,
    {
        Box::new(ChainStep::new(BoxedStepWrapper(self), next))
    }
}

/// Wrapper to make a boxed step implement Step.
struct BoxedStepWrapper<I, O>(Box<dyn Step<I, O> + Send + Sync>);

#[async_trait]
impl<I, O> Step<I, O> for BoxedStepWrapper<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        self.0.run(input, ctx).await
    }
}
