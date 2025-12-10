//! Example: Ergonomic workflow with fluent chaining and typed agents.
//!
//! This example demonstrates the new ergonomic features:
//! - Typed struct agents with explicit input/output types
//! - Fluent pipeline chaining with `.then()`
//! - Using `Workflow` for automatic metrics collection
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example ergonomic_workflow`

use gemini_structured_output::prelude::*;
use gemini_structured_output::workflow::{Step, Workflow};

// --- Data Models ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RawText {
    content: String,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Summary {
    key_points: Vec<String>,
    sentiment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EmailDraft {
    subject: String,
    body: String,
}

// --- Typed Agent Definitions ---

// Typed Struct Style: Explicit control, typed Step impl.
// This generates a struct that only implements Step<RawText, Summary>.
#[gemini_agent(
    input = "RawText",
    output = "Summary",
    system = "Summarize the text into key points and determine the overall sentiment (positive, negative, or neutral)."
)]
struct Summarizer;

// Another typed agent: Summary -> EmailDraft
#[gemini_agent(
    input = "Summary",
    output = "EmailDraft",
    system = "Draft a professional email based on this summary. The email should convey the key points concisely."
)]
struct EmailDrafter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this example");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-lite-preview-09-2025".to_string(),
        ))
        .build()?;

    // Instantiate Agents
    let summarize = Summarizer::new(client.clone());
    let draft_email = EmailDrafter::new(client.clone());

    // --- New: Fluent Workflow Chaining ---

    // Create a pipeline: RawText -> Summary -> EmailDraft
    // The compiler guarantees type safety between steps.
    // Using Workflow wrapper for automatic metrics collection
    let workflow = Workflow::new(summarize.then(draft_email)).with_name("SummaryToEmail");

    let input = RawText {
        content: "Our Q3 results show 20% revenue growth compared to last year. \
                  However, server infrastructure costs have doubled due to increased demand. \
                  Customer satisfaction scores remain high at 4.5/5. \
                  We're planning to expand to two new markets next quarter."
            .into(),
        source: "Q3 Financial Report".into(),
    };

    println!("Input: {:?}\n", input);

    // Run the entire pipeline with one call
    let (email, metrics) = workflow.run(input).await?;

    println!("=== Generated Email ===");
    println!("Subject: {}", email.subject);
    println!();
    println!("{}", email.body);

    // Display workflow metrics
    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics.steps_completed);
    println!("Total tokens: {}", metrics.total_token_count);

    Ok(())
}
