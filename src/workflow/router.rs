//! Router step for conditional workflow branching.
//!
//! This module provides `RouterStep` which delegates to different steps
//! based on a model-driven decision.

use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{GeminiStructured, Result, StructuredClient, StructuredValidator};

use super::metrics::ExecutionContext;
use super::Step;

/// A step that delegates the next action to a dispatcher based on a model decision.
///
/// The router first asks the model to make a decision based on the input,
/// then dispatches to the appropriate step based on that decision.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{RouterStep, ExecutionContext};
///
/// let router = RouterStep::new(
///     client,
///     "Decide whether to use fast or thorough analysis",
///     |decision: AnalysisType| match decision {
///         AnalysisType::Fast => Box::new(fast_analyzer),
///         AnalysisType::Thorough => Box::new(thorough_analyzer),
///     }
/// );
///
/// let ctx = ExecutionContext::new();
/// let result = router.run(input, &ctx).await?;
/// ```
pub struct RouterStep<Decision, Input, Output> {
    client: StructuredClient,
    system_prompt: String,
    dispatcher: Arc<dyn Fn(Decision) -> Box<dyn Step<Input, Output>> + Send + Sync>,
}

impl<Decision, Input, Output> RouterStep<Decision, Input, Output> {
    /// Create a new router step with a system prompt and dispatcher function.
    pub fn new(
        client: StructuredClient,
        prompt: &str,
        dispatcher: impl Fn(Decision) -> Box<dyn Step<Input, Output>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            client,
            system_prompt: prompt.to_string(),
            dispatcher: Arc::new(dispatcher),
        }
    }
}

#[async_trait]
impl<Decision, Input, Output> Step<Input, Output> for RouterStep<Decision, Input, Output>
where
    Decision: GeminiStructured + DeserializeOwned + Send + Sync + 'static,
    Decision: StructuredValidator + Serialize + Clone,
    Input: Serialize + Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    async fn run(&self, input: Input, ctx: &ExecutionContext) -> Result<Output> {
        let decision_request = self
            .client
            .request::<Decision>()
            .system(&self.system_prompt)
            .user_text(serde_json::to_string(&input)?);

        let outcome = decision_request.execute().await?;

        // Record metrics from the decision step
        ctx.record_outcome(&outcome);
        ctx.record_step();

        let next_step = (self.dispatcher)(outcome.value);
        next_step.run(input, ctx).await
    }
}
