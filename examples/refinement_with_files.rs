use std::env;

use gemini_structured_output::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct InvoiceSummary {
    vendor: String,
    total: f64,
    currency: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key =
        env::var("GEMINI_API_KEY").expect("Set GEMINI_API_KEY to run this example with the API");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini25Flash)
        .build()?;

    // Initial structured value we want to refine.
    let initial = InvoiceSummary {
        vendor: "Acme Corp".to_string(),
        total: 100.0,
        currency: "USD".to_string(),
    };

    // Create an in-memory "invoice" file to ground the refinement.
    let fake_invoice = b"Invoice\nVendor: Acme Corp\nTotal: 120.50 USD\nThank you!";
    let invoice_handle = client
        .file_manager
        .upload_bytes(fake_invoice.as_ref(), "text/plain", Some("invoice.txt"))
        .await?;

    // Dynamic context generator that can feed tabular/persona hints each iteration.
    let context_gen = |inv: &InvoiceSummary| -> String {
        format!(
            "Derived totals table:\n- Current total: {:.2} {}\n- Vendor: {}",
            inv.total, inv.currency, inv.vendor
        )
    };

    let instruction = "Reconcile the summary with the attached invoice. Use the invoice total and currency if they differ.";

    let outcome = client
        .refine(initial, instruction.to_string())
        .with_documents(vec![invoice_handle])
        .with_validator(|summary: &InvoiceSummary| {
            if summary.currency.to_uppercase() != "USD" {
                return Some("Invoice currency must remain USD for this dataset".to_string());
            }
            if summary.total <= 0.0 {
                return Some("Invoice total must be positive".to_string());
            }
            None
        })
        .with_async_validator(|summary: &InvoiceSummary| {
            let owned = summary.clone();
            async move {
                if owned.total > 10_000.0 {
                    return Some("Invoice total exceeds allowed threshold".to_string());
                }
                None
            }
        })
        .with_context_generator(context_gen)
        .execute()
        .await?;

    println!("Refined invoice summary: {:#?}", outcome.value);
    println!("Patch attempts: {}", outcome.attempts.len());
    if let Some(patch) = outcome.patch {
        println!(
            "Applied JSON Patch:\n{}",
            serde_json::to_string_pretty(&patch)?
        );
    }

    Ok(())
}
