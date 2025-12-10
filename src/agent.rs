use std::marker::PhantomData;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    error::Result,
    request::StructuredRequest,
    schema::{GeminiStructured, StructuredValidator},
    StructuredClient,
};

/// Shared state between agent steps.
pub trait WorkflowState: Clone + Send + Sync {}
impl<T: Clone + Send + Sync> WorkflowState for T {}

/// A step in a stateful workflow.
#[async_trait]
pub trait Step<S: WorkflowState>: Send + Sync {
    async fn run(&self, client: &StructuredClient, state: &mut S) -> Result<()>;
}

/// Generic extraction step that reads state, generates a typed output, and updates state.
pub struct ExtractionStep<S, Output, F, U> {
    instruction: String,
    prompt_factory: F,
    state_updater: U,
    _marker: PhantomData<(S, Output)>,
}

impl<S, Output, F, U> ExtractionStep<S, Output, F, U>
where
    S: WorkflowState,
    Output: GeminiStructured
        + StructuredValidator
        + Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static,
    F: for<'a> Fn(&'a StructuredClient, &'a S) -> StructuredRequest<'a, Output>
        + Send
        + Sync
        + 'static,
    U: Fn(&mut S, Output) + Send + Sync + 'static,
{
    pub fn new(instruction: impl Into<String>, prompt_factory: F, updater: U) -> Self {
        Self {
            instruction: instruction.into(),
            prompt_factory,
            state_updater: updater,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S, Output, F, U> Step<S> for ExtractionStep<S, Output, F, U>
where
    S: WorkflowState,
    Output: GeminiStructured
        + StructuredValidator
        + Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static,
    F: for<'a> Fn(&'a StructuredClient, &'a S) -> StructuredRequest<'a, Output>
        + Send
        + Sync
        + 'static,
    U: Fn(&mut S, Output) + Send + Sync + 'static,
{
    async fn run(&self, _client: &StructuredClient, state: &mut S) -> Result<()> {
        let mut req = (self.prompt_factory)(_client, state);
        req = req.system(self.instruction.clone());
        let outcome = req.execute().await?;
        (self.state_updater)(state, outcome.value);
        Ok(())
    }
}

/// A stateful workflow runner.
pub struct Workflow<S: WorkflowState> {
    client: StructuredClient,
    state: S,
    steps: Vec<Box<dyn Step<S>>>,
}

impl<S: WorkflowState> Workflow<S> {
    pub fn new(client: StructuredClient, state: S) -> Self {
        Self {
            client,
            state,
            steps: Vec::new(),
        }
    }

    pub fn with_step(mut self, step: Box<dyn Step<S>>) -> Self {
        self.steps.push(step);
        self
    }

    pub async fn run(mut self) -> Result<S> {
        for step in self.steps {
            step.run(&self.client, &mut self.state).await?;
        }
        Ok(self.state)
    }
}
