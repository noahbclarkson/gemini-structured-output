//! Sequential step chaining for composable workflows.
//!
//! This module provides `ChainStep` and `ChainTupleStep` which connect steps linearly,
//! where the output of the first step becomes the input of the second.

use std::sync::Arc;

use async_trait::async_trait;

use crate::Result;

use super::metrics::ExecutionContext;
use super::Step;

/// Connects two steps linearly: Output of Step A becomes Input of Step B.
///
/// This is the fundamental composition primitive for building pipelines.
/// Use the `.then()` method on any `Step` to create chains fluently.
///
/// # Type Parameters
///
/// - `I`: Input type for the first step
/// - `M`: Intermediate type (output of first, input of second)
/// - `O`: Output type of the second step
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ChainStep, Step, ExecutionContext};
///
/// let pipeline = ChainStep::new(step_a, step_b);
/// let ctx = ExecutionContext::new();
/// let result = pipeline.run(input, &ctx).await?;
/// ```
pub struct ChainStep<I, M, O> {
    first: Arc<dyn Step<I, M>>,
    second: Arc<dyn Step<M, O>>,
}

impl<I, M, O> ChainStep<I, M, O>
where
    I: Send + Sync + 'static,
    M: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Create a new chain from two steps.
    pub fn new(first: impl Step<I, M> + 'static, second: impl Step<M, O> + 'static) -> Self {
        Self {
            first: Arc::new(first),
            second: Arc::new(second),
        }
    }
}

#[async_trait]
impl<I, M, O> Step<I, O> for ChainStep<I, M, O>
where
    I: Send + Sync + 'static,
    M: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<O> {
        let intermediate = self.first.run(input, ctx).await?;
        self.second.run(intermediate, ctx).await
    }
}

/// Connects two steps, returning both the intermediate and final results as a tuple.
///
/// This is useful when you need to preserve the output of an intermediate step
/// for use in later processing or for combining results.
///
/// # Type Parameters
///
/// - `I`: Input type for the first step
/// - `M`: Intermediate type (output of first, cloned for second and result)
/// - `O`: Output type of the second step
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::workflow::{ChainTupleStep, Step, ExecutionContext};
///
/// // Get both intermediate and final results
/// let pipeline = ChainTupleStep::new(summarizer, email_drafter);
/// let ctx = ExecutionContext::new();
/// let (summary, email) = pipeline.run(raw_text, &ctx).await?;
/// ```
pub struct ChainTupleStep<I, M, O> {
    pub(crate) first: Arc<dyn Step<I, M>>,
    pub(crate) second: Arc<dyn Step<M, O>>,
}

impl<I, M, O> ChainTupleStep<I, M, O>
where
    I: Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Create a new tuple chain from two steps.
    pub fn new(first: impl Step<I, M> + 'static, second: impl Step<M, O> + 'static) -> Self {
        Self {
            first: Arc::new(first),
            second: Arc::new(second),
        }
    }
}

#[async_trait]
impl<I, M, O> Step<I, (M, O)> for ChainTupleStep<I, M, O>
where
    I: Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn run(&self, input: I, ctx: &ExecutionContext) -> Result<(M, O)> {
        let intermediate = self.first.run(input, ctx).await?;
        let output = self.second.run(intermediate.clone(), ctx).await?;
        Ok((intermediate, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::LambdaStep;

    #[tokio::test]
    async fn test_chain_step() {
        let double = LambdaStep(|x: i32| async move { Ok(x * 2) });
        let add_ten = LambdaStep(|x: i32| async move { Ok(x + 10) });

        let chain = ChainStep::new(double, add_ten);
        let ctx = ExecutionContext::new();
        let result = chain.run(5, &ctx).await.unwrap();

        // 5 * 2 = 10, then 10 + 10 = 20
        assert_eq!(result, 20);
    }

    #[tokio::test]
    async fn test_chain_type_transformation() {
        let to_string = LambdaStep(|x: i32| async move { Ok(x.to_string()) });
        let parse_len = LambdaStep(|s: String| async move { Ok(s.len()) });

        let chain = ChainStep::new(to_string, parse_len);
        let ctx = ExecutionContext::new();
        let result = chain.run(12345, &ctx).await.unwrap();

        // "12345".len() = 5
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_chain_tuple_step() {
        let double = LambdaStep(|x: i32| async move { Ok(x * 2) });
        let add_ten = LambdaStep(|x: i32| async move { Ok(x + 10) });

        let chain = ChainTupleStep::new(double, add_ten);
        let ctx = ExecutionContext::new();
        let (intermediate, result) = chain.run(5, &ctx).await.unwrap();

        // 5 * 2 = 10 (intermediate), then 10 + 10 = 20 (result)
        assert_eq!(intermediate, 10);
        assert_eq!(result, 20);
    }
}
