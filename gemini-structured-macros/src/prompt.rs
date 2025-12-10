//! GeminiPrompt derive macro for creating prompt templates.
//!
//! This macro generates a `Display` implementation that interpolates struct fields
//! into a template string.

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{DeriveInput, Fields, Lit, Meta};

/// Generate the prompt template implementation.
pub fn generate_prompt(input: DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract the template string from #[gemini(template = "...")]
    let template = match extract_template(&input) {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Extract field names for the struct
    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => fields,
            _ => {
                return quote_spanned! { input.ident.span() =>
                    compile_error!("`#[derive(GeminiPrompt)]` only supports structs with named fields");
                }
            }
        },
        _ => {
            return quote_spanned! { input.ident.span() =>
                compile_error!("`#[derive(GeminiPrompt)]` only supports structs");
            }
        }
    };

    // Parse the template and generate format arguments
    let (format_string, field_accessors) = match parse_template(&template, fields) {
        Ok(result) => result,
        Err(e) => return e,
    };

    quote! {
        impl #impl_generics std::fmt::Display for #struct_name #ty_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, #format_string, #(#field_accessors),*)
            }
        }
    }
}

/// Extract the template string from struct attributes.
fn extract_template(input: &DeriveInput) -> Result<String, TokenStream> {
    for attr in &input.attrs {
        if attr.path().is_ident("gemini") {
            let nested = match attr.parse_args::<Meta>() {
                Ok(meta) => meta,
                Err(_) => continue,
            };

            if let Meta::NameValue(nv) = nested {
                if nv.path.is_ident("template") {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(lit_str) = &expr_lit.lit {
                            return Ok(lit_str.value());
                        }
                    }
                }
            }
        }
    }

    Err(quote_spanned! { input.ident.span() =>
        compile_error!("`#[derive(GeminiPrompt)]` requires a `#[gemini(template = \"...\")]` attribute");
    })
}

/// Parse the template string and generate format string + field accessors.
///
/// Template format: "Hello {field_name}, your score is {score}."
/// Output format string: "Hello {}, your score is {}."
/// Output accessors: [self.field_name, self.score]
fn parse_template(
    template: &str,
    fields: &syn::FieldsNamed,
) -> Result<(String, Vec<TokenStream>), TokenStream> {
    let field_names: Vec<String> = fields
        .named
        .iter()
        .filter_map(|f| f.ident.as_ref().map(|i| i.to_string()))
        .collect();

    let mut format_string = String::new();
    let mut accessors: Vec<TokenStream> = Vec::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            // Check for escaped brace {{
            if chars.peek() == Some(&'{') {
                format_string.push_str("{{");
                chars.next();
                continue;
            }

            // Extract field name
            let mut field_name = String::new();
            for inner_ch in chars.by_ref() {
                if inner_ch == '}' {
                    break;
                }
                field_name.push(inner_ch);
            }

            let field_name = field_name.trim();

            // Validate field exists
            if !field_names.contains(&field_name.to_string()) {
                return Err(quote! {
                    compile_error!(concat!(
                        "Template references unknown field '",
                        #field_name,
                        "'. Available fields: ",
                        stringify!(#(#field_names),*)
                    ));
                });
            }

            format_string.push_str("{}");
            let ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
            accessors.push(quote! { self.#ident });
        } else if ch == '}' {
            // Check for escaped brace }}
            if chars.peek() == Some(&'}') {
                format_string.push_str("}}");
                chars.next();
            } else {
                format_string.push(ch);
            }
        } else {
            format_string.push(ch);
        }
    }

    Ok((format_string, accessors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_parse_template_simple() {
        let fields: syn::FieldsNamed = parse_quote! {
            { name: String, age: i32 }
        };

        let template = "Hello {name}, you are {age} years old.";
        let (format_str, accessors) = parse_template(template, &fields).unwrap();

        assert_eq!(format_str, "Hello {}, you are {} years old.");
        assert_eq!(accessors.len(), 2);
    }

    #[test]
    fn test_parse_template_escaped_braces() {
        let fields: syn::FieldsNamed = parse_quote! {
            { name: String }
        };

        let template = "JSON: {{\"name\": \"{name}\"}}";
        let (format_str, _) = parse_template(template, &fields).unwrap();

        assert_eq!(format_str, "JSON: {{\"name\": \"{}\"}}");
    }
}
