use gemini_rust::Model;
use gemini_structured_output::{ContextBuilder, StructuredClientBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Profile {
    name: String,
    title: String,
    country: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run examples");
    let model = Model::Custom("models/gemini-2.5-flash-lite-preview-09-2025".to_string());

    let client = StructuredClientBuilder::new(api_key)
        .with_model(model.clone())
        .build()?;

    let ctx = ContextBuilder::new()
        .with_system("Summarize the user text into a structured profile.")
        .add_user_text("Jane Doe is a Staff Engineer based in Canada.");

    let profile: Profile = client.generate(ctx, None).await?;
    println!("Initial profile: {profile:#?}");

    let instruction = "Promote to Principal Engineer and set country to USA.";
    let outcome = client.refine(&profile, instruction).await?;

    println!("Refined profile: {:#?}", outcome.value);
    println!("Refinement attempts: {}", outcome.attempts.len());
    for (i, att) in outcome.attempts.iter().enumerate() {
        println!(
            " Attempt {}: success={} error={:?}",
            i + 1,
            att.success,
            att.error
        );
    }

    Ok(())
}
