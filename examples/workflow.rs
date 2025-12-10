use gemini_rust::Model;
use gemini_structured_output::{ContextBuilder, StructuredClientBuilder, WorkflowStep};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResearchBrief {
    topic: String,
    bullet_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EmailDraft {
    subject: String,
    body: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run examples");
    let model = Model::Custom("models/gemini-2.5-flash-preview-09-2025".to_string());
    let client = StructuredClientBuilder::new(api_key)
        .with_model(model.clone())
        .build()?;

    // Step 1: Research a topic into structured bullets.
    let client_research = client.clone();
    let research = WorkflowStep::new("research", move |topic: String| {
        let client = client_research.clone();
        async move {
            let ctx = ContextBuilder::new()
                .with_system("Create a short research brief with 3 concise bullet points.")
                .add_user_text(format!("Topic: {topic}"));
            let brief: ResearchBrief = client.generate(ctx, None).await?;
            Ok(brief)
        }
    });

    // Step 2: Turn research into an email draft.
    let client_draft = client.clone();
    let draft_email = WorkflowStep::new("draft_email", move |brief: ResearchBrief| {
        let client = client_draft.clone();
        async move {
            let ctx = ContextBuilder::new()
                .with_system("Write a short outreach email using the research brief.")
                .add_user_text(format!(
                    "Topic: {}\nBullets:\n- {}",
                    brief.topic,
                    brief.bullet_points.join("\n- ")
                ));
            let draft: EmailDraft = client.generate(ctx, None).await?;
            Ok(draft)
        }
    });

    let brief = research.run("AI safety best practices".to_string()).await?;
    let email = draft_email.run(brief).await?;

    println!("Model: {}", model);
    println!("Email draft:\nSubject: {}\n\n{}", email.subject, email.body);

    Ok(())
}
