use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, Type};

#[derive(Debug, FromMeta)]
pub struct ToolArgs {
    #[darling(default)]
    pub name: Option<String>,
    pub description: String,
}

pub fn generate_tool(args: ToolArgs, input: ItemFn) -> TokenStream {
    let fn_name = &input.sig.ident;
    let tool_name = args.name.unwrap_or_else(|| fn_name.to_string());
    let description = args.description;
    let vis = &input.vis;
    let asyncness = &input.sig.asyncness;

    // Validate it's an async function
    if asyncness.is_none() {
        return quote! {
            compile_error!("Tool function must be async");
        };
    }

    // Extract input type (expecting exactly one argument)
    let input_type = match extract_input_type(&input) {
        Ok(ty) => ty,
        Err(err) => return err,
    };

    // Extract return type
    let return_type = match extract_return_type(&input) {
        Ok(ty) => ty,
        Err(err) => return err,
    };

    // Generate module name from function name
    let mod_name = quote::format_ident!("{}_tool", fn_name);

    quote! {
        #input

        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #vis mod #mod_name {
            use super::*;

            /// The name of this tool
            pub const NAME: &str = #tool_name;

            /// The description of this tool
            pub const DESCRIPTION: &str = #description;

            /// The input type for this tool
            pub type Input = #input_type;

            /// The output type for this tool
            pub type Output = #return_type;

            /// Register this tool with a ToolRegistry
            pub fn register(registry: gemini_structured_output::tools::ToolRegistry) -> gemini_structured_output::tools::ToolRegistry {
                registry.register_with_handler::<#input_type, #return_type, _, _>(
                    #tool_name,
                    #description,
                    super::#fn_name
                )
            }

            /// Create a tool registration closure for use with `register_tool`
            pub fn registrar() -> impl FnOnce(gemini_structured_output::tools::ToolRegistry) -> gemini_structured_output::tools::ToolRegistry {
                |registry| register(registry)
            }
        }
    }
}

fn extract_input_type(func: &ItemFn) -> Result<&Type, TokenStream> {
    let inputs = &func.sig.inputs;

    if inputs.len() != 1 {
        return Err(quote! {
            compile_error!("Tool function must take exactly one argument");
        });
    }

    match inputs.first() {
        Some(FnArg::Typed(pat_type)) => Ok(&pat_type.ty),
        Some(FnArg::Receiver(_)) => Err(quote! {
            compile_error!("Tool function cannot have self receiver");
        }),
        None => Err(quote! {
            compile_error!("Tool function must take exactly one argument");
        }),
    }
}

fn extract_return_type(func: &ItemFn) -> Result<TokenStream, TokenStream> {
    match &func.sig.output {
        syn::ReturnType::Default => Err(quote! {
            compile_error!("Tool function must have a return type of Result<T, ToolError>");
        }),
        syn::ReturnType::Type(_, ty) => {
            // Try to extract the Ok type from Result<T, E>
            if let Type::Path(type_path) = ty.as_ref() {
                if let Some(segment) = type_path.path.segments.last() {
                    if segment.ident == "Result" {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(ok_type)) = args.args.first() {
                                return Ok(quote! { #ok_type });
                            }
                        }
                    }
                }
            }
            // Fallback: assume it's just the type directly
            Ok(quote! { #ty })
        }
    }
}

/// Parse tool attributes from a list of nested meta items
pub fn parse_tool_args(attrs: &[NestedMeta]) -> darling::Result<ToolArgs> {
    ToolArgs::from_list(attrs)
}
