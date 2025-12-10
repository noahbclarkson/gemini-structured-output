use gemini_rust::Model;
use gemini_structured_output::StructuredClientBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: String,
    phone: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run examples");
    let model = Model::Custom("models/gemini-2.5-flash-lite-preview-09-2025".to_string());

    let client = StructuredClientBuilder::new(api_key)
        .with_model(model.clone())
        .build()?;

    let result = client
        .request::<Contact>()
        .system("Extract a structured contact record from the user message.")
        .user_text("Contact: Alice Example, email alice@example.com, phone 555-0101.")
        .with_thinking(512, false)
        .execute()
        .await?;

    println!("Model: {}", model);
    println!("Contact: {:#?}", result.value);
    if let Some(usage) = result.usage {
        println!(
            "Tokens -> prompt: {:?}, candidates: {:?}, cached: {:?}, total: {:?}",
            usage.prompt_token_count,
            usage.candidates_token_count,
            usage.cached_content_token_count,
            usage.total_token_count
        );
    }

    Ok(())
}
