//! Example: Outcome-based Evaluation for Forecasting.
//!
//! Run with: `GEMINI_API_KEY=... cargo run --features "evals macros" --example forecast_eval`

use gemini_structured_output::evals::LLMJudge;
use gemini_structured_output::prelude::*;

// --- 1. Data Models ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AccountContext {
    name: String,
    historical_avg: f64,
    /// e.g., "Expect a 20% bump next month"
    intent: String,
}

// The output from the LLM (The "Recipe")
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ForecastConfig {
    base_value: f64,
    modifier_percent: f64, // e.g. 0.20 for +20%
    manual_adjustment: f64,
}

// The actual calculated numbers (The "Meal")
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatedForecast {
    final_value: f64,
    delta_from_history: f64,
}

// --- 2. The Forecasting Engine (Your Domain Logic) ---

/// This represents your complex internal engine.
/// It takes the LLM's config and produces the numbers the user actually cares about.
fn run_forecast_engine(config: &ForecastConfig) -> CalculatedForecast {
    // Logic: (Base * (1 + Modifier)) + Manual
    let calculated =
        (config.base_value * (1.0 + config.modifier_percent)) + config.manual_adjustment;

    CalculatedForecast {
        final_value: calculated,
        // In a real app, this would compare against the input history
        delta_from_history: 0.0,
    }
}

// --- 3. The Generator Agent ---

#[gemini_agent(
    input = "AccountContext",
    output = "ForecastConfig",
    system = "Configure the forecast parameters to match the user's intent. \
              You can achieve the result via modifiers OR manual adjustments."
)]
struct ConfigGenerator;

// --- 4. Main Evaluation Loop ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // Generator (Flash is faster/cheaper)
    let gen_client = StructuredClientBuilder::new(&api_key)
        .with_model(Model::Gemini25Flash)
        .build()?;
    let generator = ConfigGenerator::new(gen_client);

    // Judge (Pro is smarter for reasoning)
    let judge_client = StructuredClientBuilder::new(&api_key)
        .with_model(Model::Gemini25Pro)
        .build()?;

    // The Rubric focuses on the RESULT, not how the config achieved it.
    let rubric = r#"
    1. **Accuracy**: Does the 'final_value' in the Simulation Result match the 'intent' in the Input?
    2. **Logic**: The config should be reasonable, but multiple valid configs exist.
       (e.g., A 20% modifier is the same as a manual adjustment of value * 0.20).
    3. **Pass Condition**: If the final number is within 1% of the target implied by the intent, Pass.
    "#;

    let judge = LLMJudge::new(judge_client, rubric);

    let cases = vec![
        (
            "Simple Growth".to_string(),
            AccountContext {
                name: "Revenue".to_string(),
                historical_avg: 1000.0,
                intent: "Should increase by 10%".to_string(), // Target: 1100
            },
        ),
        (
            "Flat with Adjustment".to_string(),
            AccountContext {
                name: "Expenses".to_string(),
                historical_avg: 500.0,
                intent: "Flat, plus a one-time $50 fee".to_string(), // Target: 550
            },
        ),
    ];

    let suite = EvalSuite::new("Forecast Outcome Eval").with_concurrency(2);

    let report = suite
        .run(cases, move |input| {
            let generator = generator.clone();
            let judge = judge.clone();

            async move {
                let ctx = ExecutionContext::new();

                // A. Generate the Config (The "Recipe")
                let config = generator.run(input.clone(), &ctx).await?;

                // B. EXECUTE THE ENGINE (The "Simulation")
                // This is the key step: we verify what the config actually DOES.
                let computed_result = run_forecast_engine(&config);

                // C. Judge based on Input + Config + Result
                let verdict = judge
                    .evaluate(&input, &config, Some(&computed_result))
                    .await?;

                // Return standardized outcome for the suite metrics
                let outcome = GenerationOutcome::new(config, None, vec![], None, None, 0, 0);

                Ok((outcome, verdict.pass, Some(verdict.reasoning)))
            }
        })
        .await;

    println!("{}", report);

    for result in report.results {
        if !result.passed {
            println!("Failed Case: {}", result.case_name);
        }
    }

    Ok(())
}
