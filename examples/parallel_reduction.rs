//! Example: Map-Reduce pattern with parallel processing and aggregation.
//!
//! This example demonstrates:
//! - Using `ParallelMapStep` to process multiple items concurrently
//! - Using `ReduceStep` to aggregate results into a single output
//! - Composing these steps with fluent `.then()` chaining
//! - Using `Workflow` for automatic metrics collection
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example parallel_reduction`

use gemini_structured_output::prelude::*;
use gemini_structured_output::workflow::{ParallelMapStep, ReduceStep, Step, Workflow};

// --- Data Models ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TextChunk {
    id: usize,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ChunkAnalysis {
    chunk_id: usize,
    key_topics: Vec<String>,
    sentiment: String,
    importance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct FinalReport {
    overall_sentiment: String,
    main_topics: Vec<String>,
    summary: String,
    chunk_count: usize,
}

// --- Agent for analyzing individual chunks ---

#[gemini_agent(
    input = "TextChunk",
    output = "ChunkAnalysis",
    system = "Analyze this text chunk. Extract key topics, determine sentiment (positive/negative/neutral), and assign an importance score from 0.0 to 1.0."
)]
struct ChunkAnalyzer;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this example");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-lite-preview-09-2025".to_string(),
        ))
        .build()?;

    // Sample document chunks (in a real app, these would come from a document splitter)
    let chunks = vec![
        TextChunk {
            id: 1,
            content: "Our company achieved record-breaking sales this quarter. \
                      Customer acquisition increased by 40% and retention rates improved."
                .into(),
        },
        TextChunk {
            id: 2,
            content: "Infrastructure costs have risen significantly due to cloud pricing changes. \
                      We're evaluating alternative providers to reduce expenses."
                .into(),
        },
        TextChunk {
            id: 3,
            content: "The new product launch was well-received by customers. \
                      Early adoption metrics exceed our projections by 25%."
                .into(),
        },
        TextChunk {
            id: 4,
            content:
                "Employee satisfaction surveys show room for improvement in work-life balance. \
                      HR is developing new wellness initiatives for next quarter."
                    .into(),
        },
    ];

    println!("Processing {} chunks in parallel...\n", chunks.len());

    // Step 1: Setup Map Step (Parallel Analysis)
    let analyzer = ChunkAnalyzer::new(client.clone());
    let map_step = ParallelMapStep::new(analyzer, 4); // 4 concurrent requests

    // Step 2: Setup Reduce Step (Aggregation)
    let reducer = ReduceStep::<ChunkAnalysis, FinalReport>::new(
        client.clone(),
        "Synthesize these chunk analyses into a comprehensive final report. \
         Determine the overall sentiment by weighing individual sentiments by their importance scores. \
         Consolidate and deduplicate topics. Write a brief executive summary."
    );

    // Step 3: Compose the workflow: Vec<TextChunk> -> Vec<ChunkAnalysis> -> FinalReport
    // Using Workflow wrapper for automatic metrics collection
    let workflow = Workflow::new(map_step.then(reducer)).with_name("MapReduceAnalysis");

    // Run the complete map-reduce workflow
    let (report, metrics) = workflow.run(chunks).await?;

    println!("=== Final Report ===");
    println!("Overall Sentiment: {}", report.overall_sentiment);
    println!("Chunks Analyzed: {}", report.chunk_count);
    println!("\nMain Topics:");
    for topic in &report.main_topics {
        println!("  - {}", topic);
    }
    println!("\nExecutive Summary:");
    println!("{}", report.summary);

    // Display workflow metrics
    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics.steps_completed);
    println!("Total tokens: {}", metrics.total_token_count);
    println!("  - Prompt tokens: {}", metrics.prompt_token_count);
    println!("  - Response tokens: {}", metrics.candidates_token_count);
    println!("Network attempts: {}", metrics.network_attempts);
    println!("Parse attempts: {}", metrics.parse_attempts);

    Ok(())
}
