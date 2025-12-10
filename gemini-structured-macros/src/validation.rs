use darling::{FromDeriveInput, FromField};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident};

/// Field-level validation attributes
#[derive(Debug, Clone, FromField)]
#[darling(attributes(gemini))]
pub struct FieldOpts {
    pub ident: Option<Ident>,

    /// Custom validation function: `#[gemini(validate_with = "path::to::func")]`
    /// Function should have signature: `fn(&FieldType) -> Option<String>`
    #[darling(default)]
    pub validate_with: Option<syn::Path>,

    /// Minimum value for numeric fields: `#[gemini(min = 0)]`
    #[darling(default)]
    pub min: Option<f64>,

    /// Maximum value for numeric fields: `#[gemini(max = 100)]`
    #[darling(default)]
    pub max: Option<f64>,

    /// Minimum length for string/vec fields: `#[gemini(min_len = 1)]`
    #[darling(default)]
    pub min_len: Option<usize>,

    /// Maximum length for string/vec fields: `#[gemini(max_len = 255)]`
    #[darling(default)]
    pub max_len: Option<usize>,

    /// Non-empty check for strings/vecs: `#[gemini(non_empty)]`
    #[darling(default)]
    pub non_empty: bool,

    /// Custom error message for validation failures
    #[darling(default)]
    pub error_message: Option<String>,
}

/// Struct-level validation attributes
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(gemini), supports(struct_named))]
pub struct StructOpts {
    pub ident: Ident,
    pub data: darling::ast::Data<(), FieldOpts>,

    /// Custom struct-level validation function
    #[darling(default)]
    pub validate_with: Option<syn::Path>,
}

pub fn generate_validation(input: DeriveInput) -> TokenStream {
    // Parse the input using darling
    let opts = match StructOpts::from_derive_input(&input) {
        Ok(opts) => opts,
        Err(e) => return e.write_errors(),
    };

    let struct_name = &opts.ident;

    // Extract fields
    let fields = match &opts.data {
        darling::ast::Data::Struct(fields) => fields,
        _ => {
            return quote! {
                compile_error!("GeminiValidated only supports structs with named fields");
            }
        }
    };

    // Generate validation checks for each field
    let mut field_checks = Vec::new();

    for field in fields.iter() {
        if let Some(ref ident) = field.ident {
            let field_name_str = ident.to_string();

            // Custom validator
            if let Some(ref validate_fn) = field.validate_with {
                let error_msg = field
                    .error_message
                    .clone()
                    .unwrap_or_else(|| format!("Validation failed for field '{}'", field_name_str));
                field_checks.push(quote! {
                    if let Some(err) = #validate_fn(&self.#ident) {
                        return Some(format!("{}: {}", #error_msg, err));
                    }
                });
            }

            // Min value check
            if let Some(min) = field.min {
                let error_msg = field
                    .error_message
                    .clone()
                    .unwrap_or_else(|| format!("Field '{}' must be >= {}", field_name_str, min));
                field_checks.push(quote! {
                    if (self.#ident as f64) < #min {
                        return Some(#error_msg.to_string());
                    }
                });
            }

            // Max value check
            if let Some(max) = field.max {
                let error_msg = field
                    .error_message
                    .clone()
                    .unwrap_or_else(|| format!("Field '{}' must be <= {}", field_name_str, max));
                field_checks.push(quote! {
                    if (self.#ident as f64) > #max {
                        return Some(#error_msg.to_string());
                    }
                });
            }

            // Min length check
            if let Some(min_len) = field.min_len {
                let error_msg = field.error_message.clone().unwrap_or_else(|| {
                    format!(
                        "Field '{}' must have at least {} elements",
                        field_name_str, min_len
                    )
                });
                field_checks.push(quote! {
                    if self.#ident.len() < #min_len {
                        return Some(#error_msg.to_string());
                    }
                });
            }

            // Max length check
            if let Some(max_len) = field.max_len {
                let error_msg = field.error_message.clone().unwrap_or_else(|| {
                    format!(
                        "Field '{}' must have at most {} elements",
                        field_name_str, max_len
                    )
                });
                field_checks.push(quote! {
                    if self.#ident.len() > #max_len {
                        return Some(#error_msg.to_string());
                    }
                });
            }

            // Non-empty check
            if field.non_empty {
                let error_msg = field
                    .error_message
                    .clone()
                    .unwrap_or_else(|| format!("Field '{}' must not be empty", field_name_str));
                field_checks.push(quote! {
                    if self.#ident.is_empty() {
                        return Some(#error_msg.to_string());
                    }
                });
            }
        }
    }

    // Add struct-level validation if specified
    let struct_validation = if let Some(ref validate_fn) = opts.validate_with {
        quote! {
            if let Some(err) = #validate_fn(self) {
                return Some(err);
            }
        }
    } else {
        quote! {}
    };

    // Generate the implementation
    quote! {
        impl gemini_structured_output::schema::GeminiValidator for #struct_name {
            fn gemini_validate(&self) -> Option<String> {
                #(#field_checks)*
                #struct_validation
                None
            }
        }
    }
}
