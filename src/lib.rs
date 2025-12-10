//! High-level structured output helpers built on top of `gemini-rust`.
//!
//! This crate wraps schema generation, prompt assembly, caching and refinement loops
//! into a simpler interface for application code.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use gemini_structured_output::prelude::*;
//!
//! #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//! struct Contact {
//!     name: String,
//!     email: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//!     let client = StructuredClientBuilder::new("your-api-key")
//!         .build()?;
//!
//!     let contact: Contact = client
//!         .quick_generate("Extract: John Doe, john@example.com")
//!         .await?;
//!
//!     println!("{:?}", contact);
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - **`helpers`**: Enable formatting utilities (CSV to markdown, etc.)
//! - **`macros`**: Enable procedural macros (`#[gemini_tool]`, `#[derive(GeminiValidated)]`)

pub mod adapter;
pub mod agent;
pub mod caching;
pub mod client;
pub mod context;
pub mod error;
#[cfg(feature = "evals")]
pub mod evals;
pub mod files;
#[cfg(feature = "helpers")]
pub mod helpers;
pub mod models;
pub mod patching;
pub mod request;
pub mod schema;
pub mod tools;
pub mod workflow;

pub use adapter::KeyValue;
pub use caching::CachePolicy;
pub use caching::CacheSettings;
pub use client::{
    ClientConfig, FallbackStrategy, MockHandler, MockRequest, StructuredClient,
    StructuredClientBuilder,
};
pub use context::ContextBuilder;
pub use error::{Result, ResultExt, StructuredError};
#[cfg(feature = "evals")]
pub use evals::{EvalResult, EvalSuite, SuiteReport};
pub use files::FileManager;
pub use models::{GenerationOutcome, RefinementAttempt, RefinementOutcome};
pub use patching::{ArrayPatchStrategy, PatchStrategy, RefinementConfig, RefinementEngine};
pub use request::{StreamEvent, StructuredRequest};
pub use schema::{GeminiStructured, GeminiValidator, StructuredValidator};
pub use tools::ToolRegistry;
pub use workflow::{
    BoxedStepExt, ChainStep, ChainTupleStep, ConfiguredReduceStep, ExecutionContext,
    LambdaStateStep, LambdaStep, MapStep, ParallelMapStep, ReduceStep, ReduceStepBuilder,
    ReviewStep, RouterStep, StateStep, StateWorkflow, Step, StepAdapter, WindowedContextStep,
    Workflow, WorkflowMetrics, WorkflowStep,
};

/// Prelude module for convenient imports.
///
/// ```rust
/// use gemini_structured_output::prelude::*;
/// ```
pub mod prelude {
    pub use crate::adapter::KeyValue;
    pub use crate::caching::{CachePolicy, CacheSettings};
    pub use crate::client::{
        FallbackStrategy, MockHandler, MockRequest, StructuredClient, StructuredClientBuilder,
    };
    pub use crate::context::ContextBuilder;
    pub use crate::error::{Result, ResultExt, StructuredError};
    #[cfg(feature = "evals")]
    pub use crate::evals::{EvalResult, EvalSuite, SuiteReport};
    pub use crate::models::{GenerationOutcome, RefinementOutcome};
    pub use crate::patching::{
        ArrayPatchStrategy, PatchStrategy, RefinementConfig, RefinementEngine,
    };
    pub use crate::request::{StreamEvent, StructuredRequest};
    pub use crate::schema::{GeminiStructured, GeminiValidator, StructuredValidator};
    pub use crate::tools::ToolRegistry;
    pub use crate::workflow::{
        BoxedStepExt, ChainStep, ChainTupleStep, ConfiguredReduceStep, ExecutionContext,
        LambdaStateStep, LambdaStep, MapStep, ParallelMapStep, ReduceStep, ReduceStepBuilder,
        ReviewStep, RouterStep, StateStep, StateWorkflow, Step, StepAdapter, WindowedContextStep,
        Workflow, WorkflowMetrics, WorkflowStep,
    };

    // Re-export commonly used external types
    pub use gemini_rust::Model;
    pub use schemars::JsonSchema;
    pub use serde::{Deserialize, Serialize};

    // Re-export macros when the feature is enabled
    #[cfg(feature = "macros")]
    pub use gemini_structured_macros::{gemini_agent, gemini_tool, GeminiPrompt, GeminiValidated};
}

#[cfg(feature = "helpers")]
pub use helpers::{
    bullet_list, code_block, collapsible, csv_to_markdown, csv_to_markdown_with_options,
    format_currency, format_number, json_array_to_markdown, key_value, key_value_block,
    numbered_list, truncate_text, CsvError, CsvOptions, JsonTableError, TableAlignment,
};

#[cfg(feature = "macros")]
pub use gemini_structured_macros::{gemini_agent, gemini_tool, GeminiPrompt, GeminiValidated};
