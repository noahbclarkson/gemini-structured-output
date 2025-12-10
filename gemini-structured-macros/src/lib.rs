//! Procedural macros for `gemini-structured-output`.
//!
//! This crate provides:
//! - `#[gemini_tool]`: Attribute macro for defining tool functions
//! - `#[gemini_agent]`: Attribute macro for defining agents (struct or functional style)
//! - `#[derive(GeminiValidated)]`: Derive macro for adding validation to structs
//! - `#[derive(GeminiPrompt)]`: Derive macro for creating prompt templates

mod agent;
mod prompt;
mod tools;
mod validation;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

/// Attribute macro for defining Gemini tool functions.
///
/// This macro transforms an async function into a tool that can be registered
/// with a `ToolRegistry`. The function must:
/// - Be async
/// - Take exactly one argument that implements `JsonSchema + Serialize + DeserializeOwned`
/// - Return `Result<T, ToolError>` where `T` implements `JsonSchema + Serialize`
///
/// # Arguments
///
/// - `description` (required): A description of what the tool does
/// - `name` (optional): Override the tool name (defaults to function name)
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_output::prelude::*;
/// use gemini_structured_macros::gemini_tool;
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, JsonSchema)]
/// struct StockRequest {
///     symbol: String,
/// }
///
/// #[derive(Debug, Serialize, Deserialize, JsonSchema)]
/// struct StockPrice {
///     symbol: String,
///     price: f64,
/// }
///
/// #[gemini_tool(description = "Look up the current price of a stock")]
/// async fn get_stock_price(args: StockRequest) -> Result<StockPrice, ToolError> {
///     Ok(StockPrice {
///         symbol: args.symbol,
///         price: 150.0,
///     })
/// }
///
/// // Register the tool:
/// let registry = ToolRegistry::new().register_tool(get_stock_price_tool::registrar());
/// ```
#[proc_macro_attribute]
pub fn gemini_tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let attr_args = match darling::ast::NestedMeta::parse_meta_list(args.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(darling::Error::from(e).write_errors()),
    };

    let tool_args = match tools::parse_tool_args(&attr_args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let func = parse_macro_input!(input as syn::ItemFn);
    tools::generate_tool(tool_args, func).into()
}

/// Attribute macro for quickly turning a struct into a Gemini agent.
///
/// This macro generates a struct that holds a `StructuredClient` and implements
/// the `workflow::Step` trait, allowing the agent to be composed into workflows.
///
/// # Generic Style
///
/// When no `input`/`output` types are specified, the agent works with any compatible types:
///
/// ```rust,ignore
/// #[gemini_agent(system = "Extract accounts from text.")]
/// struct ListExtractor;
///
/// // Use with any compatible input/output types:
/// let accounts: AccountList = extractor.run(pdf_text).await?;
/// ```
///
/// # Typed Style
///
/// When `input` and `output` are specified, the agent only implements Step for those types:
///
/// ```rust,ignore
/// #[gemini_agent(
///     input = "Summary",
///     output = "EmailDraft",
///     system = "Draft an email based on this summary."
/// )]
/// struct EmailDrafter;
///
/// // Type-safe: only works with Summary -> EmailDraft
/// let email: EmailDraft = drafter.run(summary).await?;
/// ```
///
/// # Fluent Chaining
///
/// Typed agents can be chained together using `.then()`:
///
/// ```rust,ignore
/// let pipeline = summarizer.then(email_drafter);
/// let email = pipeline.run(raw_text).await?;
/// ```
///
/// # Arguments
///
/// - `system` (required): The system prompt for the agent.
/// - `model` (optional): A model hint; configure your `StructuredClient` with the same model.
/// - `input` (optional): Explicit input type as a string, e.g., `"MyInputType"`.
/// - `output` (optional): Explicit output type as a string, e.g., `"MyOutputType"`.
#[proc_macro_attribute]
pub fn gemini_agent(args: TokenStream, input: TokenStream) -> TokenStream {
    let attr_args = match darling::ast::NestedMeta::parse_meta_list(args.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(darling::Error::from(e).write_errors()),
    };

    let agent_args = match agent::parse_agent_args(&attr_args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input = parse_macro_input!(input as DeriveInput);
    agent::generate_agent(agent_args, input).into()
}

/// Derive macro for implementing `StructuredValidator` with declarative validation rules.
///
/// This macro generates a `StructuredValidator` implementation based on attributes
/// placed on struct fields. It allows you to specify validation rules declaratively
/// rather than implementing `validate()` manually.
///
/// # Field Attributes
///
/// - `#[gemini(validate_with = "path::to::func")]`: Custom validation function
/// - `#[gemini(min = N)]`: Minimum value for numeric fields
/// - `#[gemini(max = N)]`: Maximum value for numeric fields
/// - `#[gemini(min_len = N)]`: Minimum length for string/vec fields
/// - `#[gemini(max_len = N)]`: Maximum length for string/vec fields
/// - `#[gemini(non_empty)]`: Require non-empty string/vec
/// - `#[gemini(error_message = "...")]`: Custom error message
///
/// # Struct Attributes
///
/// - `#[gemini(validate_with = "path::to::func")]`: Struct-level validation function
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_macros::GeminiValidated;
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// fn validate_even(value: &i32) -> Option<String> {
///     if *value % 2 != 0 {
///         Some("value must be even".to_string())
///     } else {
///         None
///     }
/// }
///
/// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, GeminiValidated)]
/// struct UserProfile {
///     #[gemini(non_empty, max_len = 50)]
///     name: String,
///
///     #[gemini(min = 0, max = 150)]
///     age: i32,
///
///     #[gemini(validate_with = "validate_even", error_message = "Score validation")]
///     score: i32,
///
///     #[gemini(min_len = 1)]
///     tags: Vec<String>,
/// }
/// ```
#[proc_macro_derive(GeminiValidated, attributes(gemini))]
pub fn derive_gemini_validated(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    validation::generate_validation(input).into()
}

/// Derive macro for creating prompt templates from structs.
///
/// This macro generates a `Display` implementation that interpolates struct fields
/// into a template string, making it easy to construct prompts from structured data.
///
/// # Example
///
/// ```rust,ignore
/// use gemini_structured_macros::GeminiPrompt;
///
/// #[derive(GeminiPrompt)]
/// #[gemini(template = "Analyze the {document_type} titled '{title}' and provide {analysis_type} analysis.")]
/// struct AnalysisRequest {
///     document_type: String,
///     title: String,
///     analysis_type: String,
/// }
///
/// let request = AnalysisRequest {
///     document_type: "report".to_string(),
///     title: "Q3 Financials".to_string(),
///     analysis_type: "sentiment".to_string(),
/// };
///
/// // Automatically generates the prompt string
/// let prompt = request.to_string();
/// // => "Analyze the report titled 'Q3 Financials' and provide sentiment analysis."
/// ```
#[proc_macro_derive(GeminiPrompt, attributes(gemini))]
pub fn derive_gemini_prompt(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    prompt::generate_prompt(input).into()
}
