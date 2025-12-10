//! Example: branching + parallel agentic workflow with strongly-typed steps.
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features macros --example agentic_workflow`

use gemini_structured_output::gemini_agent;
use gemini_structured_output::prelude::*;
use gemini_structured_output::workflow::{
    ExecutionContext, LambdaStep, ParallelMapStep, RouterStep, Step,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct PdfChunk {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AccountStub {
    id: String,
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AccountList {
    accounts: Vec<AccountStub>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AccountDetail {
    id: String,
    name: String,
    balance: f64,
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ValidationResult {
    id: String,
    ok: bool,
    reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct DetailPrompt {
    account: AccountStub,
    context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum NextAction {
    Validate,
    SkipValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Decision {
    action: NextAction,
}

// --- Agents declared via macro (derive client + Step impls) ---

#[gemini_agent(system = "Extract all accounts (id + name) mentioned in the text.")]
struct ListExtractor;

#[gemini_agent(
    system = "Extract full details for the referenced account using the provided context.",
    temperature = 0.4
)]
struct DetailExtractor;

#[gemini_agent(
    system = "Compare new details to previous record and mark ok=true when consistent.",
    temperature = 0.0,
    retries = 5
)]
struct Validator;

#[tokio::main]
async fn main() -> Result<()> {
    // Enable logging output for this example.
    tracing_subscriber::fmt::init();

    let api_key = std::env::var("GEMINI_API_KEY")
        .expect("set GEMINI_API_KEY to run this example against the API");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-lite-preview-09-2025".to_string(),
        ))
        .build()?;

    // Mock PDF text + old DB; replace with real data in your app.
    let pdf_text = "Account 1234 (Acme Corp) shows current balance 42.50. \
                    Account 5678 (Beta LLC) shows balance 99.0."
        .to_string();
    let previous_records = vec![AccountDetail {
        id: "1234".into(),
        name: "Acme Corp".into(),
        balance: 40.0,
        status: "active".into(),
    }];

    let (results, metrics) = run_account_processing(client, pdf_text, previous_records).await?;

    println!("Final validation results: {results:#?}");

    // Display workflow metrics
    println!("\n=== Workflow Metrics ===");
    println!("Steps completed: {}", metrics.steps_completed);
    println!("Total tokens: {}", metrics.total_token_count);

    Ok(())
}

async fn run_account_processing(
    client: StructuredClient,
    pdf_text: String,
    historical: Vec<AccountDetail>,
) -> Result<(Vec<ValidationResult>, WorkflowMetrics)> {
    // Create an execution context to track metrics across all steps
    let ctx = ExecutionContext::new();

    // Step 1: list accounts
    let list_agent = ListExtractor::new(client.clone());
    let list: AccountList = list_agent
        .run(
            PdfChunk {
                text: pdf_text.clone(),
            },
            &ctx,
        )
        .await?;

    // Step 2: parallel detail extraction
    let detail_inputs: Vec<DetailPrompt> = list
        .accounts
        .iter()
        .cloned()
        .map(|account| DetailPrompt {
            account,
            context: pdf_text.clone(),
        })
        .collect();

    let detail_worker = DetailExtractor::new(client.clone());
    let parallel = ParallelMapStep::new(detail_worker, 4);
    let details: Vec<AccountDetail> = parallel.run(detail_inputs, &ctx).await?;

    // Step 3: conditional validation via router
    let router = RouterStep::new(
        client.clone(),
        "Decide whether to validate. If balances look unusual, choose Validate.",
        move |decision: Decision| {
            let c = client.clone();
            let historical = historical.clone();
            match decision.action {
                NextAction::Validate => Box::new(LambdaStep(move |items: Vec<AccountDetail>| {
                    let c = c.clone();
                    let historical = historical.clone();
                    async move {
                        let inner_ctx = ExecutionContext::new();
                        let mut out = Vec::with_capacity(items.len());
                        for item in items {
                            let previous = historical.iter().find(|p| p.id == item.id);
                            let payload = json!({ "new": item, "previous": previous });
                            let res: ValidationResult =
                                Validator::new(c.clone()).run(payload, &inner_ctx).await?;
                            out.push(res);
                        }
                        Ok(out)
                    }
                }))
                    as Box<dyn Step<Vec<AccountDetail>, Vec<ValidationResult>>>,
                NextAction::SkipValidation => {
                    Box::new(LambdaStep(|items: Vec<AccountDetail>| async move {
                        Ok(items
                            .into_iter()
                            .map(|item| ValidationResult {
                                id: item.id,
                                ok: true,
                                reason: None,
                            })
                            .collect())
                    }))
                        as Box<dyn Step<Vec<AccountDetail>, Vec<ValidationResult>>>
                }
            }
        },
    );

    let results = router.run(details, &ctx).await?;
    let metrics = ctx.snapshot();

    Ok((results, metrics))
}
