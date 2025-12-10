//! Agent macro implementation for generating typed workflow steps.

use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{DeriveInput, Fields};

/// Arguments for the `#[gemini_agent]` attribute macro.
#[derive(Debug, FromMeta)]
pub struct AgentArgs {
    /// Optional model hint for the agent.
    #[darling(default)]
    pub model: Option<syn::LitStr>,
    /// The system prompt for the agent.
    pub system: syn::LitStr,
    /// Optional explicit input type for typed Step implementation (as string to parse).
    #[darling(default)]
    pub input: Option<syn::LitStr>,
    /// Optional explicit output type for typed Step implementation (as string to parse).
    #[darling(default)]
    pub output: Option<syn::LitStr>,
    /// Optional temperature override for this agent.
    #[darling(default)]
    pub temperature: Option<f32>,
    /// Optional retry override for this agent.
    #[darling(default)]
    pub retries: Option<usize>,
}

/// Parse agent arguments from attribute metadata.
pub fn parse_agent_args(attrs: &[NestedMeta]) -> darling::Result<AgentArgs> {
    AgentArgs::from_list(attrs)
}

/// Generate code for a struct-based agent.
///
/// Supports two modes:
/// 1. Generic mode (no input/output specified): Implements Step<I, O> for any compatible types
/// 2. Typed mode (input/output specified): Implements Step<Input, Output> for specific types
pub fn generate_agent(args: AgentArgs, input: DeriveInput) -> TokenStream {
    let DeriveInput {
        attrs,
        vis,
        ident: struct_name,
        generics,
        data,
    } = input;

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let system_prompt = args.system;
    let model_hint = match args.model {
        Some(model) => quote! { Some(#model) },
        None => quote! { Option::<&str>::None },
    };

    let fields_valid = match &data {
        syn::Data::Struct(data) => matches!(data.fields, Fields::Unit),
        _ => false,
    };

    if !fields_valid {
        let span = struct_name.span();
        return quote_spanned! { span =>
            compile_error!("`#[gemini_agent]` expects a unit struct like `struct MyAgent;`");
        };
    }

    let params = StepGenParams {
        struct_name: &struct_name,
        impl_generics: &impl_generics,
        ty_generics: &ty_generics,
        where_clause,
        system_prompt: &system_prompt,
        model_hint: &model_hint,
        temperature: args.temperature,
        retries: args.retries,
    };

    // Check if we have explicit input/output types
    let step_impl = match (args.input, args.output) {
        (Some(input_str), Some(output_str)) => {
            // Parse the type strings
            let input_type: syn::Type = match syn::parse_str(&input_str.value()) {
                Ok(t) => t,
                Err(e) => {
                    return syn::Error::new(input_str.span(), format!("Invalid input type: {}", e))
                        .to_compile_error()
                }
            };
            let output_type: syn::Type = match syn::parse_str(&output_str.value()) {
                Ok(t) => t,
                Err(e) => {
                    return syn::Error::new(
                        output_str.span(),
                        format!("Invalid output type: {}", e),
                    )
                    .to_compile_error()
                }
            };

            // Typed mode: generate a concrete Step implementation
            generate_typed_step_impl(&params, &input_type, &output_type)
        }
        (None, None) => {
            // Generic mode: generate a generic Step implementation
            generate_generic_step_impl(&params)
        }
        _ => {
            let span = struct_name.span();
            return quote_spanned! { span =>
                compile_error!("`#[gemini_agent]` requires both `input` and `output` to be specified, or neither");
            };
        }
    };

    quote! {
        #(#attrs)*
        #[derive(Clone)]
        #vis struct #struct_name {
            client: gemini_structured_output::StructuredClient,
        }

        impl #impl_generics #struct_name #ty_generics #where_clause {
            pub fn new(client: gemini_structured_output::StructuredClient) -> Self {
                Self { client }
            }
        }

        #step_impl
    }
}

/// Parameters for generating Step implementations.
struct StepGenParams<'a> {
    struct_name: &'a syn::Ident,
    impl_generics: &'a syn::ImplGenerics<'a>,
    ty_generics: &'a syn::TypeGenerics<'a>,
    where_clause: Option<&'a syn::WhereClause>,
    system_prompt: &'a syn::LitStr,
    model_hint: &'a TokenStream,
    temperature: Option<f32>,
    retries: Option<usize>,
}

/// Generate a generic Step implementation (for any I, O).
fn generate_generic_step_impl(params: &StepGenParams) -> TokenStream {
    let StepGenParams {
        struct_name,
        ty_generics,
        where_clause,
        system_prompt,
        model_hint,
        temperature,
        retries,
        ..
    } = params;

    let temp_setter = temperature
        .map(|t| quote! { request = request.temperature(#t); })
        .unwrap_or_else(|| quote! {});
    let retry_setter = retries
        .map(|r| quote! { request = request.retries(#r); })
        .unwrap_or_else(|| quote! {});
    quote! {
        #[async_trait::async_trait]
        impl<I, O> gemini_structured_output::workflow::Step<I, O> for #struct_name #ty_generics #where_clause
        where
            I: serde::Serialize + Send + Sync + 'static,
            O: gemini_structured_output::GeminiStructured
                + serde::de::DeserializeOwned
                + serde::Serialize
                + std::clone::Clone
                + Send
                + Sync
                + 'static,
        {
            async fn run(
                &self,
                input: I,
                ctx: &gemini_structured_output::workflow::ExecutionContext,
            ) -> gemini_structured_output::Result<O> {
                let input_text = serde_json::to_string(&input)?;
                tracing::debug!(
                    target: "gemini_agent",
                    agent = stringify!(#struct_name),
                    input_bytes = input_text.len(),
                    "Agent started"
                );
                let model_hint = #model_hint;
                if let Some(model) = model_hint {
                    tracing::trace!(
                        target: "gemini_structured_output::agent",
                        model,
                        "agent requested model hint; configure StructuredClient accordingly"
                    );
                }

                let mut request = self
                    .client
                    .request::<O>()
                    .system(#system_prompt)
                    .user_text(input_text);

                #temp_setter
                #retry_setter

                let outcome = request.execute().await?;

                // Automatic metric recording
                ctx.record_outcome(&outcome);
                ctx.record_step();

                Ok(outcome.value)
            }
        }
    }
}

/// Generate a typed Step implementation for specific Input/Output types.
fn generate_typed_step_impl(
    params: &StepGenParams,
    input_type: &syn::Type,
    output_type: &syn::Type,
) -> TokenStream {
    let StepGenParams {
        struct_name,
        impl_generics,
        ty_generics,
        where_clause,
        system_prompt,
        model_hint,
        temperature,
        retries,
    } = params;

    let temp_setter = temperature
        .map(|t| quote! { request = request.temperature(#t); })
        .unwrap_or_else(|| quote! {});
    let retry_setter = retries
        .map(|r| quote! { request = request.retries(#r); })
        .unwrap_or_else(|| quote! {});
    quote! {
        #[async_trait::async_trait]
        impl #impl_generics gemini_structured_output::workflow::Step<#input_type, #output_type> for #struct_name #ty_generics #where_clause
        {
            async fn run(
                &self,
                input: #input_type,
                ctx: &gemini_structured_output::workflow::ExecutionContext,
            ) -> gemini_structured_output::Result<#output_type> {
                let input_text = serde_json::to_string(&input)?;
                tracing::debug!(
                    target: "gemini_agent",
                    agent = stringify!(#struct_name),
                    input_bytes = input_text.len(),
                    "Agent started"
                );
                let model_hint = #model_hint;
                if let Some(model) = model_hint {
                    tracing::trace!(
                        target: "gemini_structured_output::agent",
                        model,
                        "agent requested model hint; configure StructuredClient accordingly"
                    );
                }

                let mut request = self
                    .client
                    .request::<#output_type>()
                    .system(#system_prompt)
                    .user_text(input_text);

                #temp_setter
                #retry_setter

                let outcome = request.execute().await?;

                // Automatic metric recording
                ctx.record_outcome(&outcome);
                ctx.record_step();

                Ok(outcome.value)
            }
        }
    }
}
