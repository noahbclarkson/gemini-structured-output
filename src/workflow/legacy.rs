//! Legacy workflow step implementation.
//!
//! This module provides backward-compatible workflow types for existing code.

use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;

use super::metrics::ExecutionContext;
use super::Step;
use crate::error::Result;

/// Pinned future returned by a workflow step.
pub type WorkflowFuture<Output> = Pin<Box<dyn Future<Output = Result<Output>> + Send>>;

/// Type alias to simplify the action signature.
pub type WorkflowAction<Input, Output> = Box<dyn Fn(Input) -> WorkflowFuture<Output> + Send + Sync>;

/// A composable workflow step used to chain multiple model calls.
pub struct WorkflowStep<Input, Output> {
    pub name: String,
    pub action: WorkflowAction<Input, Output>,
}

impl<Input, Output> WorkflowStep<Input, Output> {
    pub fn new<F, Fut>(name: &str, f: F) -> Self
    where
        F: Fn(Input) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output>> + Send + 'static,
    {
        Self {
            name: name.to_string(),
            action: Box::new(move |input| Box::pin(f(input))),
        }
    }

    pub async fn run(&self, input: Input) -> Result<Output> {
        self.execute(input).await
    }

    async fn execute(&self, input: Input) -> Result<Output> {
        tracing::info!("Running workflow step '{}'", self.name);
        (self.action)(input).await
    }
}

#[async_trait]
impl<Input, Output> Step<Input, Output> for WorkflowStep<Input, Output>
where
    Input: Send + 'static,
    Output: Send + 'static,
{
    async fn run(&self, input: Input, _ctx: &ExecutionContext) -> Result<Output> {
        self.execute(input).await
    }
}
