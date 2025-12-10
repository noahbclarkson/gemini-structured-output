//! Aggregation/reduction step for map-reduce workflows.
//!
//! This module provides `ReduceStep` which takes a `Vec<T>` and synthesizes
//! a single structured output, useful for aggregating results from parallel
//! operations like `ParallelMapStep`.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};

use crate::{schema::GeminiStructured, Result, StructuredClient};

use super::metrics::ExecutionContext;
use super::Step;

/// Summarizes or aggregates a list of inputs into a single structured output.
///
/// This step is commonly used after a `ParallelMapStep` to combine multiple
/// results into a final aggregated form.
///
/// # Type Parameters
///
/// - `InputItem`: The type of each item in the input vector
/// - `Output`: The aggregated output type
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ReduceStep, ExecutionContext};
///
/// // Aggregate multiple analysis results into a final report
/// let reducer = ReduceStep::<Analysis, FinalReport>::new(
///     client,
///     "Merge these analyses into a comprehensive final report."
/// );
///
/// let analyses = vec![analysis1, analysis2, analysis3];
/// let ctx = ExecutionContext::new();
/// let report = reducer.run(analyses, &ctx).await?;
/// ```
pub struct ReduceStep<InputItem, Output> {
    client: StructuredClient,
    system_prompt: String,
    _marker: PhantomData<(InputItem, Output)>,
}

impl<InputItem, Output> ReduceStep<InputItem, Output> {
    /// Create a new reduce step with a system prompt describing the aggregation.
    ///
    /// # Arguments
    ///
    /// - `client`: The `StructuredClient` to use for API calls
    /// - `system_prompt`: Instructions for how to aggregate the items
    pub fn new(client: StructuredClient, system_prompt: impl Into<String>) -> Self {
        Self {
            client,
            system_prompt: system_prompt.into(),
            _marker: PhantomData,
        }
    }

    /// Create a reduce step with a custom user prompt format.
    ///
    /// The format string should contain `{}` which will be replaced with
    /// the serialized input items.
    pub fn with_format(
        client: StructuredClient,
        system_prompt: impl Into<String>,
    ) -> ReduceStepBuilder<InputItem, Output> {
        ReduceStepBuilder {
            client,
            system_prompt: system_prompt.into(),
            user_format: None,
            _marker: PhantomData,
        }
    }
}

/// Builder for `ReduceStep` with additional configuration options.
pub struct ReduceStepBuilder<InputItem, Output> {
    client: StructuredClient,
    system_prompt: String,
    user_format: Option<String>,
    _marker: PhantomData<(InputItem, Output)>,
}

impl<InputItem, Output> ReduceStepBuilder<InputItem, Output> {
    /// Set a custom user prompt format.
    ///
    /// Use `{}` as a placeholder for the serialized input data.
    pub fn user_format(mut self, format: impl Into<String>) -> Self {
        self.user_format = Some(format.into());
        self
    }

    /// Build the final `ReduceStep`.
    pub fn build(self) -> ConfiguredReduceStep<InputItem, Output> {
        ConfiguredReduceStep {
            client: self.client,
            system_prompt: self.system_prompt,
            user_format: self
                .user_format
                .unwrap_or_else(|| "Aggregate the following data:\n{}".to_string()),
            _marker: PhantomData,
        }
    }
}

/// A `ReduceStep` with custom formatting options.
pub struct ConfiguredReduceStep<InputItem, Output> {
    client: StructuredClient,
    system_prompt: String,
    user_format: String,
    _marker: PhantomData<(InputItem, Output)>,
}

#[async_trait]
impl<InputItem, Output> Step<Vec<InputItem>, Output> for ReduceStep<InputItem, Output>
where
    InputItem: Serialize + Send + Sync + 'static,
    Output: GeminiStructured + Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn run(&self, items: Vec<InputItem>, ctx: &ExecutionContext) -> Result<Output> {
        let input_text = serde_json::to_string_pretty(&items)?;

        let outcome = self
            .client
            .request::<Output>()
            .system(&self.system_prompt)
            .user_text(format!("Aggregate the following data:\n{}", input_text))
            .execute()
            .await?;

        // Record metrics from this step
        ctx.record_outcome(&outcome);
        ctx.record_step();

        Ok(outcome.value)
    }
}

#[async_trait]
impl<InputItem, Output> Step<Vec<InputItem>, Output> for ConfiguredReduceStep<InputItem, Output>
where
    InputItem: Serialize + Send + Sync + 'static,
    Output: GeminiStructured + Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn run(&self, items: Vec<InputItem>, ctx: &ExecutionContext) -> Result<Output> {
        let input_text = serde_json::to_string_pretty(&items)?;
        let user_prompt = self.user_format.replace("{}", &input_text);

        let outcome = self
            .client
            .request::<Output>()
            .system(&self.system_prompt)
            .user_text(user_prompt)
            .execute()
            .await?;

        // Record metrics from this step
        ctx.record_outcome(&outcome);
        ctx.record_step();

        Ok(outcome.value)
    }
}
