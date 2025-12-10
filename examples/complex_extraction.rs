use gemini_rust::Model;
use gemini_structured_output::{StructuredClientBuilder, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct InvoiceData {
    invoice_number: String,
    total_amount: f64,
    /// The 3-letter currency code (e.g., "USD", "EUR").
    #[schemars(
        description = "The 3-letter ISO currency code (e.g. 'USD'). Do not return an object."
    )]
    currency: String,
    line_items: Vec<LineItem>,
    /// Exchange rate to USD for the given currency.
    usd_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LineItem {
    description: String,
    quantity: i32,
    unit_price: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CurrencyLookup {
    code: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CurrencyResult {
    symbol: String,
    /// Exchange rate to USD (demo values).
    usd_rate: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable not set");
    let model = Model::Custom("models/gemini-2.5-flash-preview-09-2025".to_string());
    let client = StructuredClientBuilder::new(api_key)
        .with_model(model.clone())
        .build()?;

    let tools = ToolRegistry::new().register_with_handler::<CurrencyLookup, CurrencyResult, _, _>(
        "lookup_currency",
        "Return currency symbol",
        |args| async move {
            let symbol = match args.code.as_str() {
                "USD" => "$",
                "EUR" => "€",
                "GBP" => "£",
                "NZD" => "$",
                _ => "?",
            };
            let usd_rate = match args.code.as_str() {
                "USD" => 1.0,
                "EUR" => 1.08,
                "GBP" => 1.25,
                "NZD" => 0.60,
                _ => 1.0,
            };
            Ok(CurrencyResult {
                symbol: symbol.to_string(),
                usd_rate,
            })
        },
    );

    let pdf_path = "documents/invoice.pdf";
    // Ensure the file exists or create a dummy one for the example if needed
    if !std::path::Path::new(pdf_path).exists() {
        println!("Note: invoice.pdf not found, creating a dummy file for demonstration.");
        std::fs::create_dir_all("documents")?;
        std::fs::write(pdf_path, b"Invoice #12345\nItem: Widget A - $10.00 x 5\nItem: Widget B - $20.00 x 2\nTotal: $90.00\nCurrency: EUR")?;
    }

    let outcome = client
        .request::<InvoiceData>()
        .system(
            "Extract invoice data in the source currency. Include usd_rate for that currency; \
             do not convert amounts.",
        )
        .add_file_path(pdf_path)
        .await?
        .user_text("Extract this invoice, include currency and usd_rate fields (no conversion).")
        .with_tools(tools)
        .refine_with(
            "Convert amounts to USD using usd_rate. Set currency to USD. \
             Recompute total_amount from line items (quantity * unit_price) in USD.",
        )
        .execute()
        .await?;

    println!("Model: {}", model);
    println!("Final Invoice: {:#?}", outcome.value);
    Ok(())
}
