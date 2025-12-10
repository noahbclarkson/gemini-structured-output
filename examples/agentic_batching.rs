//! Example: Windowed Parallel Processing with Context Sharing and Final Review.
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example agentic_batching`

use gemini_structured_output::gemini_agent;
use gemini_structured_output::prelude::*;
use gemini_structured_output::workflow::{
    ExecutionContext, LambdaStep, ReviewStep, Step, WindowedContextStep,
};

// --- Data Models ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ClientStub {
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ClientList {
    clients: Vec<ClientStub>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ClientProfile {
    name: String,
    /// Industry or sector (e.g., Tech, Retail)
    industry: Option<String>,
    /// Estimated annual revenue if mentioned
    revenue_est: Option<String>,
    /// Key contact person if mentioned
    key_contact: Option<String>,
    /// A generic catch-all for messy notes
    notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ProfileBatch {
    profiles: Vec<ClientProfile>,
}

// --- Agents ---

#[gemini_agent(system = "Extract a list of all client names mentioned in the text.")]
struct NameExtractor;

#[gemini_agent(
    system = "You are a detailed researcher. Given a list of client names and a source document, \
              extract detailed profiles for ONLY the clients in the provided list. \
              If a specific field (like revenue) is missing in the text, leave it null."
)]
struct BatchProcessor;

// --- Main Workflow ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-lite-preview-09-2025".to_string(),
        ))
        .build()?;

    // Create an execution context to track metrics across all steps
    let ctx = ExecutionContext::new();

    // 1. The "Hidden Info" Document
    let raw_document = r#"
        INTERNAL MEMO: Q4 PROSPECTS

        We met with **Alpha Dynamics** yesterday. They are moving fast in the Aerospace sector.
        John Smith is the lead there. They mentioned a budget of roughly $5M/year.

        Then there is **Beta Foods**. A small retail chain. Not much budget, maybe $500k.
        Contact is Sarah.

        **Gamma Solutions** is tricky. Tech sector. They are undergoing a merger.
        No revenue figures disclosed yet, but they want to start next month.

        Finally, **Delta Corp**. Big manufacturing player. Revenue $50M+.
        Talk to Mike regarding the implementation.
    "#
    .to_string();

    println!("--- Step 1: Extracting Client List ---");

    // Run Agent 1: Extract Names
    let name_extractor = NameExtractor::new(client.clone());
    let client_list: ClientList = name_extractor.run(raw_document.clone(), &ctx).await?;

    println!(
        "Found clients: {:?}",
        client_list
            .clients
            .iter()
            .map(|c| &c.name)
            .collect::<Vec<_>>()
    );

    println!("\n--- Step 2: Running Windowed Parallel Agents (Batch Size: 2) ---");

    // Define the worker for the windowed step.
    let client_for_batches = client.clone();
    let batch_worker = LambdaStep(move |(batch, context): (Vec<ClientStub>, String)| {
        let agent = BatchProcessor::new(client_for_batches.clone());
        async move {
            tracing::info!("Agent processing batch of size: {}", batch.len());

            #[derive(Serialize)]
            struct BatchInput {
                target_clients: Vec<ClientStub>,
                source_text: String,
            }

            // LambdaStep doesn't have access to ctx, so we create a local one
            let inner_ctx = ExecutionContext::new();
            let result: ProfileBatch = agent
                .run(
                    BatchInput {
                        target_clients: batch,
                        source_text: context,
                    },
                    &inner_ctx,
                )
                .await?;

            Ok(result.profiles)
        }
    });

    // Create the Windowed Step
    let windowed_step = WindowedContextStep::new(batch_worker, 2, 2);

    let profiles: Vec<ClientProfile> = windowed_step
        .run((client_list.clients, raw_document.clone()), &ctx)
        .await?;

    println!("Extracted {} profiles.", profiles.len());
    for p in &profiles {
        println!(
            " - {} ({:?})",
            p.name,
            p.industry.as_deref().unwrap_or("Unknown")
        );
    }

    println!("\n--- Step 3: Final Review & Validation ---");

    let reviewer = ReviewStep::new(
        client.clone(),
        "Review the extracted profiles against the source text. \
         Ensure revenue figures and contact names are accurate. \
         Fix any hallucinations or missing data.",
    );

    let final_profiles: Vec<ClientProfile> = reviewer.run((profiles, raw_document), &ctx).await?;

    println!("\n=== Final Validated Output ===");
    println!("{}", serde_json::to_string_pretty(&final_profiles)?);

    // Display workflow metrics
    let metrics = ctx.snapshot();
    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics.steps_completed);
    println!("Total tokens: {}", metrics.total_token_count);
    println!("  - Prompt tokens: {}", metrics.prompt_token_count);
    println!("  - Response tokens: {}", metrics.candidates_token_count);

    Ok(())
}
