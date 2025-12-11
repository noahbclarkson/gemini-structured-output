//! Automatic instrumentation for workflow steps.
//!
//! This module provides `InstrumentedStep` which wraps any step to automatically
//! emit `StepStart` and `StepEnd` events, enabling comprehensive workflow tracing
//! without manual instrumentation.

use std::time::Instant;

use async_trait::async_trait;

use crate::Result;

use super::events::WorkflowEvent;
use super::metrics::ExecutionContext;
use super::Step;

/// A wrapper that automatically instruments a step with start/end event emission.
///
/// `InstrumentedStep` wraps any step and automatically:
/// - Emits a `StepStart` event when execution begins
/// - Emits a `StepEnd` event on success (with duration)
/// - Emits an `Error` event on failure
///
/// This eliminates the need for manual event emission in step implementations
/// and ensures consistent tracing across all workflow steps.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{Step, ExecutionContext};
///
/// // Using .named() combinator (recommended)
/// let pipeline = summarizer.named("Summarize")
///     .then(drafter.named("DraftEmail"));
///
/// let ctx = ExecutionContext::new();
/// let result = pipeline.run(input, &ctx).await?;
///
/// // All step executions are automatically traced
/// for trace in ctx.trace_snapshot() {
///     println!("{:?}", trace);
/// }
/// ```
pub struct InstrumentedStep<S> {
    /// The wrapped step.
    pub inner: S,
    /// The name used for tracing events.
    pub name: String,
}

impl<S> InstrumentedStep<S> {
    /// Create a new instrumented step with the given name.
    pub fn new(inner: S, name: impl Into<String>) -> Self {
        Self {
            inner,
            name: name.into(),
        }
    }

    /// Get the name of this instrumented step.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a reference to the inner step.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Unwrap and return the inner step.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

#[async_trait]
impl<S, I, O> Step<I, O> for InstrumentedStep<S>
where
    S: Step<I, O>,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        ctx.emit(WorkflowEvent::StepStart {
            step_name: self.name.clone(),
            input_type: std::any::type_name::<I>().to_string(),
        });

        let start = Instant::now();
        let result = self.inner.run(input, ctx).await;
        let duration = start.elapsed().as_millis();

        match &result {
            Ok(_) => {
                ctx.emit(WorkflowEvent::StepEnd {
                    step_name: self.name.clone(),
                    duration_ms: duration,
                });
            }
            Err(e) => {
                ctx.emit(WorkflowEvent::Error {
                    step_name: self.name.clone(),
                    message: e.to_string(),
                });
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{LambdaStep, WorkflowEvent};

    #[tokio::test]
    async fn test_instrumented_step_emits_start_and_end() {
        let step = LambdaStep(|x: i32| async move { Ok(x * 2) });
        let instrumented = InstrumentedStep::new(step, "Double");

        let ctx = ExecutionContext::new();
        let result = instrumented.run(5, &ctx).await.unwrap();

        assert_eq!(result, 10);

        let traces = ctx.trace_snapshot();
        assert_eq!(traces.len(), 2);

        assert!(matches!(
            &traces[0].event,
            WorkflowEvent::StepStart { step_name, .. } if step_name == "Double"
        ));

        assert!(matches!(
            &traces[1].event,
            WorkflowEvent::StepEnd { step_name, .. }
            if step_name == "Double"
        ));
    }

    #[tokio::test]
    async fn test_instrumented_step_emits_error_on_failure() {
        let step = LambdaStep(|_x: i32| async move {
            Err::<i32, _>(crate::StructuredError::Validation("test error".to_string()))
        });
        let instrumented = InstrumentedStep::new(step, "Failing");

        let ctx = ExecutionContext::new();
        let result: Result<i32> = instrumented.run(5, &ctx).await;

        assert!(result.is_err());

        let traces = ctx.trace_snapshot();
        assert_eq!(traces.len(), 2);

        assert!(matches!(
            &traces[0].event,
            WorkflowEvent::StepStart { step_name, .. } if step_name == "Failing"
        ));

        assert!(matches!(
            &traces[1].event,
            WorkflowEvent::Error { step_name, message }
            if step_name == "Failing" && message.contains("test error")
        ));
    }

    #[tokio::test]
    async fn test_instrumented_step_captures_input_type() {
        let step = LambdaStep(|s: String| async move { Ok(s.len()) });
        let instrumented = InstrumentedStep::new(step, "StringLen");

        let ctx = ExecutionContext::new();
        let _ = instrumented.run("hello".to_string(), &ctx).await;

        let traces = ctx.trace_snapshot();
        if let WorkflowEvent::StepStart { input_type, .. } = &traces[0].event {
            assert!(input_type.contains("String"));
        } else {
            panic!("Expected StepStart event");
        }
    }
}
