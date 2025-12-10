use std::{future::Future, marker::PhantomData, sync::Arc};

use async_trait::async_trait;

use crate::{workflow::Step, Result};

use super::{ExecutionContext, WorkflowMetrics};

/// A workflow step that operates on a shared mutable state.
#[async_trait]
pub trait StateStep<S>: Send + Sync {
    async fn run(&self, state: &mut S, ctx: &ExecutionContext) -> Result<()>;
}

/// Simple wrapper to build a [`StateStep`] from an async function or closure.
pub struct LambdaStateStep<F, S> {
    func: Arc<F>,
    _marker: PhantomData<S>,
}

impl<F, Fut, S> LambdaStateStep<F, S>
where
    F: Fn(&mut S, &ExecutionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
    S: Send + Sync + 'static,
{
    pub fn new(func: F) -> Self {
        Self {
            func: Arc::new(func),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Fut, S> StateStep<S> for LambdaStateStep<F, S>
where
    F: Fn(&mut S, &ExecutionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
    S: Send + Sync + 'static,
{
    async fn run(&self, state: &mut S, ctx: &ExecutionContext) -> Result<()> {
        (self.func)(state, ctx).await
    }
}

/// Adapter that allows a regular [`Step`] to participate in a stateful workflow.
pub struct StepAdapter<I, O, S, G, Set> {
    inner: Arc<dyn Step<I, O>>,
    getter: G,
    setter: Set,
    _marker: PhantomData<S>,
}

impl<I, O, S, G, Set> StepAdapter<I, O, S, G, Set>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    S: Send + Sync + 'static,
    G: Fn(&S) -> I + Send + Sync + 'static,
    Set: Fn(&mut S, O) + Send + Sync + 'static,
{
    pub fn new(step: impl Step<I, O> + 'static, getter: G, setter: Set) -> Self {
        Self {
            inner: Arc::new(step),
            getter,
            setter,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, S, G, Set> StateStep<S> for StepAdapter<I, O, S, G, Set>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    S: Send + Sync + 'static,
    G: Fn(&S) -> I + Send + Sync + 'static,
    Set: Fn(&mut S, O) + Send + Sync + 'static,
{
    async fn run(&self, state: &mut S, ctx: &ExecutionContext) -> Result<()> {
        let input = (self.getter)(state);
        let output = self.inner.run(input, ctx).await?;
        (self.setter)(state, output);
        Ok(())
    }
}

/// Workflow runner that carries a mutable state object across steps.
pub struct StateWorkflow<S> {
    state: S,
    steps: Vec<Box<dyn StateStep<S>>>,
    name: Option<String>,
}

impl<S> StateWorkflow<S>
where
    S: Send + Sync + 'static,
{
    /// Create a new stateful workflow with the provided initial state.
    pub fn new(state: S) -> Self {
        Self {
            state,
            steps: Vec::new(),
            name: None,
        }
    }

    /// Name the workflow for logging/metrics.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a custom state step.
    pub fn step(mut self, step: impl StateStep<S> + 'static) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Add a closure-based step.
    pub fn step_fn<F, Fut>(self, func: F) -> Self
    where
        F: Fn(&mut S, &ExecutionContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.step(LambdaStateStep::new(func))
    }

    /// Add a regular [`Step`] with getter/setter adapters.
    pub fn with_adapter<I, O, G, Set>(
        self,
        step: impl Step<I, O> + 'static,
        getter: G,
        setter: Set,
    ) -> Self
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
        G: Fn(&S) -> I + Send + Sync + 'static,
        Set: Fn(&mut S, O) + Send + Sync + 'static,
    {
        self.step(StepAdapter::new(step, getter, setter))
    }

    /// Run the workflow, returning the final state and metrics.
    pub async fn run(self) -> Result<(S, WorkflowMetrics)> {
        let ctx = ExecutionContext::new();
        self.run_with_context(ctx).await
    }

    /// Run with a provided execution context to aggregate metrics across workflows.
    pub async fn run_with_context(self, ctx: ExecutionContext) -> Result<(S, WorkflowMetrics)> {
        let mut state = self.state;

        if let Some(name) = &self.name {
            tracing::info!("Starting state workflow: {}", name);
        }

        for step in &self.steps {
            if let Err(err) = step.run(&mut state, &ctx).await {
                ctx.record_failure(err.to_string());
                if let Some(name) = &self.name {
                    tracing::error!("State workflow '{name}' failed: {err}");
                }
                return Err(err);
            }
            ctx.record_step();
        }

        let metrics = ctx.snapshot();

        if let Some(name) = &self.name {
            tracing::info!(
                "State workflow '{}' completed. Steps: {}, Tokens: {}",
                name,
                metrics.steps_completed,
                metrics.total_token_count
            );
        }

        Ok((state, metrics))
    }
}
