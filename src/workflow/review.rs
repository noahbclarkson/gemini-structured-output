//! Review step for refining data against a context.
//!
//! This module provides `ReviewStep` which reviews data against a provided context
//! and refines it if needed.

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};

use crate::Result;
use crate::{
    schema::{GeminiStructured, StructuredValidator},
    StructuredClient,
};

use super::metrics::ExecutionContext;
use super::Step;

/// A workflow step that reviews data against a provided context and refines it if needed.
pub struct ReviewStep<Data> {
    client: StructuredClient,
    instruction: String,
    _marker: std::marker::PhantomData<Data>,
}

impl<Data> ReviewStep<Data> {
    /// Create a new review step with an instruction for refinement.
    pub fn new(client: StructuredClient, instruction: impl Into<String>) -> Self {
        Self {
            client,
            instruction: instruction.into(),
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<Data, Context> Step<(Data, Context), Data> for ReviewStep<Data>
where
    Data: GeminiStructured
        + StructuredValidator
        + Serialize
        + DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static,
    Context: std::fmt::Display + Send + Sync + 'static,
{
    async fn run(&self, (data, context): (Data, Context), ctx: &ExecutionContext) -> Result<Data> {
        let prompt = format!("{}\n\nCONTEXT:\n{}", self.instruction, context);
        let outcome = self
            .client
            .refine(data, prompt)
            .execute()
            .await?;

        // Record step completion
        ctx.record_step();

        Ok(outcome.value)
    }
}
