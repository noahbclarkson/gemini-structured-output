//! Example: Workflow Observability and Human-in-the-Loop
//!
//! This example demonstrates the new workflow observability features:
//! - Structured execution tracing with `WorkflowEvent` and `TraceEntry`
//! - The `.tap()` combinator for side-effect inspection
//! - The `.named()` combinator for automatic instrumentation
//! - `CheckpointStep` for human-in-the-loop workflows
//! - `BatchStep` for chunked parallel processing
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example observability`

use gemini_structured_output::prelude::*;

// --- Data Models ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Article {
    title: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Summary {
    headline: String,
    key_points: Vec<String>,
    word_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SentimentAnalysis {
    sentiment: String,
    confidence: f32,
    reasoning: String,
}

// --- Agent Definitions ---

#[gemini_agent(
    input = "Article",
    output = "Summary",
    system = "Summarize this article. Provide a compelling headline, \
              3-5 key points, and the approximate word count."
)]
struct Summarizer;

#[gemini_agent(
    input = "Summary",
    output = "SentimentAnalysis",
    system = "Analyze the sentiment of this summary. Determine if the overall \
              tone is positive, negative, neutral, or mixed. Provide a confidence \
              score (0.0-1.0) and explain your reasoning."
)]
struct SentimentAnalyzer;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini25Flash)
        .build()?;

    // === Example 1: Named Steps with Automatic Tracing ===
    println!("=== Example 1: Automatic Instrumentation with .named() ===\n");

    let summarizer = Summarizer::new(client.clone());
    let sentiment_analyzer = SentimentAnalyzer::new(client.clone());

    // Chain steps with automatic start/end event emission
    let pipeline = summarizer
        .named("Summarize")
        .then(sentiment_analyzer.named("AnalyzeSentiment"));

    let article = Article {
        title: "Tech Industry Sees Record Growth".into(),
        content: "The technology sector reported unprecedented growth this quarter, \
                  with cloud services and AI leading the charge. Major companies \
                  announced significant investments in infrastructure, while startups \
                  attracted record venture capital funding. Analysts predict continued \
                  momentum through the end of the year."
            .into(),
    };

    let ctx = ExecutionContext::new();
    println!("Processing: {}\n", article.title);

    let sentiment = pipeline.run(article.clone(), &ctx).await?;

    println!("Result: {} (confidence: {:.0}%)", sentiment.sentiment, sentiment.confidence * 100.0);
    println!("Reasoning: {}\n", sentiment.reasoning);

    // View the execution trace
    println!("=== Execution Trace ===");
    for entry in ctx.trace_snapshot() {
        match &entry.event {
            WorkflowEvent::StepStart { step_name, input_type } => {
                println!("[{}ms] START: {} (input: {})", entry.timestamp % 100000, step_name, input_type);
            }
            WorkflowEvent::StepEnd { step_name, duration_ms } => {
                println!("[{}ms] END: {} (took {}ms)", entry.timestamp % 100000, step_name, duration_ms);
            }
            WorkflowEvent::Error { step_name, message } => {
                println!("[{}ms] ERROR: {} - {}", entry.timestamp % 100000, step_name, message);
            }
            WorkflowEvent::Artifact { step_name, key, data } => {
                println!("[{}ms] ARTIFACT: {}.{} = {}", entry.timestamp % 100000, step_name, key, data);
            }
        }
    }

    // === Example 2: Using .tap() for Side-Effect Inspection ===
    println!("\n\n=== Example 2: Side-Effect Inspection with .tap() ===\n");

    let summarizer2 = Summarizer::new(client.clone());
    let sentiment_analyzer2 = SentimentAnalyzer::new(client.clone());

    // Use tap to inspect and emit artifacts at each stage
    let pipeline_with_tap = summarizer2
        .named("Summarize")
        .tap(|summary: &Summary, ctx: &ExecutionContext| {
            // Log the summary details
            println!("  [tap] Summary generated: {} key points", summary.key_points.len());
            // Emit a custom artifact to the trace
            ctx.emit_artifact("Summarize", "key_point_count", &summary.key_points.len());
        })
        .then(sentiment_analyzer2.named("AnalyzeSentiment"))
        .tap(|sentiment: &SentimentAnalysis, ctx: &ExecutionContext| {
            println!("  [tap] Sentiment: {} ({:.0}% confidence)", sentiment.sentiment, sentiment.confidence * 100.0);
            ctx.emit_artifact("AnalyzeSentiment", "confidence", &sentiment.confidence);
        });

    let ctx2 = ExecutionContext::new();

    let article2 = Article {
        title: "Market Volatility Concerns Investors".into(),
        content: "Financial markets experienced significant turbulence this week, \
                  with major indices dropping sharply before partial recovery. \
                  Concerns about inflation and interest rates continue to weigh \
                  on investor sentiment. Some analysts recommend caution while \
                  others see buying opportunities."
            .into(),
    };

    println!("Processing: {}", article2.title);
    let result2 = pipeline_with_tap.run(article2, &ctx2).await?;
    println!("\nFinal sentiment: {}", result2.sentiment);

    // Show artifacts in trace
    println!("\n=== Artifacts Recorded ===");
    for entry in ctx2.trace_snapshot() {
        if let WorkflowEvent::Artifact { step_name, key, data } = &entry.event {
            println!("  {}.{} = {}", step_name, key, data);
        }
    }

    // === Example 3: Conditional Checkpoint for Human Review ===
    println!("\n\n=== Example 3: Conditional Checkpoint (Human-in-the-Loop) ===\n");

    let summarizer3 = Summarizer::new(client.clone());
    let sentiment_analyzer3 = SentimentAnalyzer::new(client.clone());

    // Create a pipeline that pauses for review when confidence is low
    let pipeline_with_checkpoint = summarizer3
        .named("Summarize")
        .then(sentiment_analyzer3.named("AnalyzeSentiment"))
        .then(ConditionalCheckpointStep::new(
            "LowConfidenceReview",
            |sentiment: &SentimentAnalysis| sentiment.confidence < 0.7,
        ));

    let ctx3 = ExecutionContext::new();

    let article3 = Article {
        title: "Mixed Signals in Economic Data".into(),
        content: "Economic indicators released today present a complex picture. \
                  Employment figures exceeded expectations while consumer spending \
                  showed signs of slowing. Manufacturing output was flat, but \
                  service sector growth remained strong. Economists are divided \
                  on what this means for the overall trajectory."
            .into(),
    };

    println!("Processing: {}", article3.title);
    println!("(Pipeline will pause if sentiment confidence < 70%)\n");

    match pipeline_with_checkpoint.run(article3, &ctx3).await {
        Ok(sentiment) => {
            println!("Pipeline completed without checkpoint");
            println!("Sentiment: {} (confidence: {:.0}%)", sentiment.sentiment, sentiment.confidence * 100.0);
        }
        Err(StructuredError::Checkpoint { step_name, data }) => {
            println!("*** CHECKPOINT TRIGGERED: {} ***\n", step_name);
            println!("Intermediate data saved for human review:");
            println!("{}\n", serde_json::to_string_pretty(&data)?);

            // In a real application, you would:
            // 1. Save 'data' to a database
            // 2. Notify a human reviewer
            // 3. Resume the pipeline later with modified data

            println!("In production, this data would be queued for human review.");
            println!("The workflow can be resumed after approval.");
        }
        Err(e) => return Err(e),
    }

    // === Example 4: Manual Event Emission ===
    println!("\n\n=== Example 4: Manual Event Emission ===\n");

    let ctx4 = ExecutionContext::new();

    // You can emit events directly for custom instrumentation
    ctx4.emit(WorkflowEvent::StepStart {
        step_name: "CustomProcessing".to_string(),
        input_type: "String".to_string(),
    });

    // Simulate some work
    println!("Performing custom processing...");
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Emit an artifact
    ctx4.emit_artifact("CustomProcessing", "items_processed", &42);

    ctx4.emit(WorkflowEvent::StepEnd {
        step_name: "CustomProcessing".to_string(),
        duration_ms: 50,
    });

    println!("\n=== Custom Trace ===");
    for entry in ctx4.trace_snapshot() {
        println!("{:?}", entry.event);
    }

    // === Example 5: Metrics Summary ===
    println!("\n\n=== Example 5: Combined Metrics Summary ===\n");

    let metrics = ctx.snapshot();
    println!("Pipeline 1 Metrics:");
    println!("  Steps completed: {}", metrics.steps_completed);
    println!("  Total tokens: {}", metrics.total_token_count);

    let metrics2 = ctx2.snapshot();
    println!("\nPipeline 2 Metrics:");
    println!("  Steps completed: {}", metrics2.steps_completed);
    println!("  Total tokens: {}", metrics2.total_token_count);

    let metrics3 = ctx3.snapshot();
    println!("\nPipeline 3 Metrics:");
    println!("  Steps completed: {}", metrics3.steps_completed);
    println!("  Total tokens: {}", metrics3.total_token_count);

    Ok(())
}
