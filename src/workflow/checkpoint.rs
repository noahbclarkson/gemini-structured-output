//! Human-in-the-loop checkpoint step for workflow pausing.
//!
//! This module provides `CheckpointStep` which intentionally halts workflow
//! execution, allowing human review or modification of intermediate data
//! before resuming.

use async_trait::async_trait;
use serde::Serialize;

use crate::{Result, StructuredError};

use super::events::WorkflowEvent;
use super::metrics::ExecutionContext;
use super::Step;

/// A step that intentionally halts execution, returning the input data.
///
/// `CheckpointStep` is used for human-in-the-loop workflows where execution
/// should pause at a specific point for human review, approval, or modification.
///
/// When executed, this step:
/// 1. Serializes the input data to JSON
/// 2. Emits a `StepEnd` event to the trace log
/// 3. Returns a `Checkpoint` error containing the serialized data
///
/// The workflow can be "resumed" by:
/// 1. Extracting the data from the checkpoint error
/// 2. Allowing human modification
/// 3. Running the remaining pipeline steps with the modified data
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{CheckpointStep, Step, ExecutionContext};
/// use gemini_structured_output::StructuredError;
///
/// // Build pipeline with a checkpoint for human review
/// let pipeline = generator
///     .then(CheckpointStep::new("ReviewDraft"))
///     .then(saver);
///
/// let ctx = ExecutionContext::new();
/// match pipeline.run(input, &ctx).await {
///     Ok(_) => println!("Workflow completed automatically"),
///     Err(StructuredError::Checkpoint { step_name, data }) => {
///         println!("Paused at '{}'. Data: {}", step_name, data);
///         // Save 'data' for human review
///         // User can later resume by running the 'saver' step with modified data
///     }
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub struct CheckpointStep<T> {
    name: String,
    _marker: std::marker::PhantomData<T>,
}

impl<T> CheckpointStep<T> {
    /// Create a new checkpoint step with the given name.
    ///
    /// The name is used to identify the checkpoint in error messages and traces.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the checkpoint name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl<T> Step<T, T> for CheckpointStep<T>
where
    T: Serialize + Send + Sync + 'static,
{
    async fn run(&self, input: T, ctx: &ExecutionContext) -> Result<T> {
        let data = serde_json::to_value(&input).map_err(StructuredError::Json)?;

        ctx.emit(WorkflowEvent::StepEnd {
            step_name: self.name.clone(),
            duration_ms: 0,
        });

        Err(StructuredError::Checkpoint {
            step_name: self.name.clone(),
            data,
        })
    }
}

/// A conditional checkpoint that only triggers when a predicate is true.
///
/// This is useful when you want to pause execution only under certain conditions,
/// such as when confidence is low or when specific validation fails.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ConditionalCheckpointStep, Step};
///
/// // Only checkpoint if confidence is below threshold
/// let conditional_checkpoint = ConditionalCheckpointStep::new(
///     "LowConfidence",
///     |result: &AnalysisResult| result.confidence < 0.8,
/// );
///
/// let pipeline = analyzer
///     .then(conditional_checkpoint)
///     .then(saver);
/// ```
pub struct ConditionalCheckpointStep<T, F> {
    name: String,
    predicate: F,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F> ConditionalCheckpointStep<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    /// Create a new conditional checkpoint.
    ///
    /// The checkpoint will only trigger when `predicate` returns `true`.
    pub fn new(name: impl Into<String>, predicate: F) -> Self {
        Self {
            name: name.into(),
            predicate,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<T, F> Step<T, T> for ConditionalCheckpointStep<T, F>
where
    T: Serialize + Send + Sync + 'static,
    F: Fn(&T) -> bool + Send + Sync,
{
    async fn run(&self, input: T, ctx: &ExecutionContext) -> Result<T> {
        if (self.predicate)(&input) {
            let data = serde_json::to_value(&input).map_err(StructuredError::Json)?;

            ctx.emit(WorkflowEvent::StepEnd {
                step_name: self.name.clone(),
                duration_ms: 0,
            });

            Err(StructuredError::Checkpoint {
                step_name: self.name.clone(),
                data,
            })
        } else {
            Ok(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestData {
        value: i32,
        text: String,
    }

    #[tokio::test]
    async fn test_checkpoint_returns_error_with_data() {
        let checkpoint = CheckpointStep::<TestData>::new("TestCheckpoint");
        let ctx = ExecutionContext::new();

        let input = TestData {
            value: 42,
            text: "hello".to_string(),
        };

        let result = checkpoint.run(input.clone(), &ctx).await;

        match result {
            Err(StructuredError::Checkpoint { step_name, data }) => {
                assert_eq!(step_name, "TestCheckpoint");
                let recovered: TestData = serde_json::from_value(data).unwrap();
                assert_eq!(recovered, input);
            }
            _ => panic!("Expected Checkpoint error"),
        }
    }

    #[tokio::test]
    async fn test_conditional_checkpoint_triggers_when_true() {
        let checkpoint =
            ConditionalCheckpointStep::new("HighValue", |data: &TestData| data.value > 50);

        let ctx = ExecutionContext::new();
        let input = TestData {
            value: 100,
            text: "high".to_string(),
        };

        let result = checkpoint.run(input, &ctx).await;
        assert!(matches!(result, Err(StructuredError::Checkpoint { .. })));
    }

    #[tokio::test]
    async fn test_conditional_checkpoint_passes_when_false() {
        let checkpoint =
            ConditionalCheckpointStep::new("HighValue", |data: &TestData| data.value > 50);

        let ctx = ExecutionContext::new();
        let input = TestData {
            value: 30,
            text: "low".to_string(),
        };

        let result = checkpoint.run(input.clone(), &ctx).await;
        assert_eq!(result.unwrap(), input);
    }

    #[tokio::test]
    async fn test_checkpoint_emits_trace_event() {
        let checkpoint = CheckpointStep::<TestData>::new("TracedCheckpoint");
        let ctx = ExecutionContext::new();

        let input = TestData {
            value: 1,
            text: "test".to_string(),
        };

        let _ = checkpoint.run(input, &ctx).await;

        let traces = ctx.trace_snapshot();
        assert!(!traces.is_empty());

        let has_end_event = traces.iter().any(|t| {
            matches!(
                &t.event,
                WorkflowEvent::StepEnd { step_name, .. } if step_name == "TracedCheckpoint"
            )
        });
        assert!(has_end_event);
    }
}
