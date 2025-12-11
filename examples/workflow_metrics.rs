//! Example: Complete Workflow with Metrics, Map Combinator, and Observability.
//!
//! This example demonstrates the full workflow abstraction:
//! - Using `Workflow` for automatic metrics collection
//! - Using `.map()` combinator for inline data transformations
//! - Using `StateWorkflow` to preserve intermediate results without tuple nesting
//! - Token usage tracking across multi-step pipelines
//! - Error handling and failure tracking
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example workflow_metrics`

use gemini_structured_output::prelude::*;
use gemini_structured_output::workflow::{ExecutionContext, StateWorkflow, Step, Workflow};

// --- Data Models ---

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
struct Document {
    title: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Analysis {
    summary: String,
    key_themes: Vec<String>,
    sentiment: String,
    word_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EnrichedAnalysis {
    analysis: Analysis,
    importance_score: f32,
    reading_time_minutes: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ExecutiveBrief {
    headline: String,
    key_takeaways: Vec<String>,
    recommended_action: String,
}

#[derive(Debug, Clone, Default)]
struct AnalysisState {
    doc: Document,
    analysis: Option<Analysis>,
    enriched: Option<EnrichedAnalysis>,
    brief: Option<ExecutiveBrief>,
}

// --- Agent Definitions ---

#[gemini_agent(
    input = "Document",
    output = "Analysis",
    system = "Analyze this document. Provide a concise summary, extract key themes, \
              determine overall sentiment (positive/negative/neutral/mixed), \
              and estimate word count."
)]
struct DocumentAnalyzer;

#[gemini_agent(
    input = "EnrichedAnalysis",
    output = "ExecutiveBrief",
    system = "Create an executive brief from this enriched analysis. \
              Write a compelling headline, list 3-5 key takeaways, \
              and recommend a concrete action item."
)]
struct BriefGenerator;

// --- Helper Functions ---

fn calculate_importance_score(analysis: &Analysis) -> f32 {
    let theme_score = (analysis.key_themes.len() as f32 * 0.1).min(0.5);
    let sentiment_score = match analysis.sentiment.to_lowercase().as_str() {
        "positive" => 0.3,
        "negative" => 0.4,
        "mixed" => 0.2,
        _ => 0.1,
    };
    theme_score + sentiment_score
}

fn estimate_reading_time(word_count: usize) -> f32 {
    (word_count as f32 / 200.0).max(1.0)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this example");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-lite-preview-09-2025".to_string(),
        ))
        .build()?;

    // --- Example 1: Simple Workflow with Metrics ---
    println!("=== Example 1: Simple Workflow ===\n");

    let analyzer = DocumentAnalyzer::new(client.clone());
    let brief_gen = BriefGenerator::new(client.clone());

    // Build pipeline with map combinator:
    // Document -> Analysis -> EnrichedAnalysis -> ExecutiveBrief
    let pipeline = analyzer
        .map(|analysis: Analysis| {
            // Inline calculation: enrich the analysis with derived data
            let importance_score = calculate_importance_score(&analysis);
            let reading_time_minutes = estimate_reading_time(analysis.word_count);
            EnrichedAnalysis {
                analysis,
                importance_score,
                reading_time_minutes,
            }
        })
        .then(brief_gen);

    // Wrap in Workflow for automatic metrics
    let workflow = Workflow::new(pipeline).with_name("DocumentToBrief");

    let document = Document {
        title: "Q4 Strategic Initiative".into(),
        content: "Our company is launching a major digital transformation initiative. \
                  Key areas include cloud migration, AI integration, and process automation. \
                  Early pilots show 30% efficiency gains. Employee adoption has been positive, \
                  though training requirements remain significant. Budget allocation is on track, \
                  with projected ROI within 18 months."
            .into(),
    };

    println!("Processing document: {}\n", document.title);

    let (brief, metrics) = workflow.run(document).await?;

    println!("=== Executive Brief ===");
    println!("Headline: {}", brief.headline);
    println!("\nKey Takeaways:");
    for takeaway in &brief.key_takeaways {
        println!("  - {}", takeaway);
    }
    println!("\nRecommended Action: {}", brief.recommended_action);

    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics.steps_completed);
    println!("Total tokens: {}", metrics.total_token_count);
    println!("  - Prompt tokens: {}", metrics.prompt_token_count);
    println!("  - Response tokens: {}", metrics.candidates_token_count);
    println!("Network attempts: {}", metrics.network_attempts);
    println!("Parse correction attempts: {}", metrics.parse_attempts);
    if !metrics.failures.is_empty() {
        println!("Failures recorded: {}", metrics.failures.len());
        for failure in &metrics.failures {
            println!("  - {}", failure);
        }
    }

    // --- Example 2: Stateful workflow to avoid tuple hell ---
    println!("\n\n=== Example 2: Blackboard State (no tuples) ===\n");

    let analyzer2 = DocumentAnalyzer::new(client.clone());
    let brief_gen2 = BriefGenerator::new(client.clone());

    let document2 = Document {
        title: "Customer Feedback Summary".into(),
        content: "Customer satisfaction scores improved by 15% this month. \
                  Main praise focused on faster response times and helpful support staff. \
                  Areas for improvement include mobile app stability and documentation clarity. \
                  NPS score reached 72, our highest ever. Churn rate decreased to 2.1%."
            .into(),
    };

    println!("Processing document: {}\n", document2.title);

    let state_workflow = StateWorkflow::new(AnalysisState {
        doc: document2.clone(),
        ..Default::default()
    })
    .with_name("AnalysisAndBrief")
    .with_adapter(
        analyzer2,
        |state: &AnalysisState| state.doc.clone(),
        |state, analysis| {
            let importance_score = calculate_importance_score(&analysis);
            let reading_time_minutes = estimate_reading_time(analysis.word_count);
            state.enriched = Some(EnrichedAnalysis {
                analysis: analysis.clone(),
                importance_score,
                reading_time_minutes,
            });
            state.analysis = Some(analysis);
        },
    )
    .with_adapter(
        brief_gen2,
        |state: &AnalysisState| {
            state
                .enriched
                .clone()
                .expect("analysis step must populate state before generating brief")
        },
        |state, brief| {
            state.brief = Some(brief);
        },
    );

    let (final_state, metrics2) = state_workflow.run().await?;
    let enriched = final_state
        .enriched
        .expect("stateful workflow populates enriched analysis");
    let brief2 = final_state
        .brief
        .expect("stateful workflow populates executive brief");

    println!("=== Enriched Analysis ===");
    println!("Summary: {}", enriched.analysis.summary);
    println!("Sentiment: {}", enriched.analysis.sentiment);
    println!("Key Themes: {:?}", enriched.analysis.key_themes);
    println!("Importance Score: {:.2}", enriched.importance_score);
    println!(
        "Estimated Reading Time: {:.1} minutes",
        enriched.reading_time_minutes
    );

    println!("\n=== Executive Brief ===");
    println!("Headline: {}", brief2.headline);
    println!("Recommended Action: {}", brief2.recommended_action);

    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics2.steps_completed);
    println!("Total tokens: {}", metrics2.total_token_count);

    // --- Example 3: Direct ExecutionContext Usage ---
    println!("\n\n=== Example 3: Manual Context for Fine-Grained Control ===\n");

    let ctx = ExecutionContext::new();

    let analyzer3 = DocumentAnalyzer::new(client.clone());

    let doc3 = Document {
        title: "Technical Incident Report".into(),
        content: "Database failover occurred at 14:32 UTC. \
                  Root cause: network partition in primary datacenter. \
                  Recovery time: 4 minutes. No data loss confirmed. \
                  Affected services: 3. Customer impact: minimal."
            .into(),
    };

    println!("Processing: {}", doc3.title);

    // Run step directly with context
    let analysis3: Analysis = analyzer3.run(doc3, &ctx).await?;

    println!("\nAnalysis complete:");
    println!("  Summary: {}", analysis3.summary);
    println!("  Sentiment: {}", analysis3.sentiment);

    // Get snapshot of metrics at any point
    let snapshot = ctx.snapshot();
    println!("\n=== Context Metrics Snapshot ===");
    println!("Steps so far: {}", snapshot.steps_completed);
    println!("Tokens used: {}", snapshot.total_token_count);

    // Context can be reused for multiple operations
    ctx.record_failure("Simulated failure for demonstration");

    let final_metrics = ctx.snapshot();
    println!("\n=== Final Context State ===");
    println!("Total steps: {}", final_metrics.steps_completed);
    println!("Failures: {:?}", final_metrics.failures);

    Ok(())
}
