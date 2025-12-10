//! Financial Forecast Example
//!
//! This example demonstrates supporting `HashMap` structured output via the
//! adapter pattern (map ↔ list of key/value objects) and then refining it.

use futures::StreamExt;
use gemini_rust::{GenerationConfig, Model};
use gemini_structured_output::{
    adapter::KeyValue, ArrayPatchStrategy, GeminiStructured, StreamEvent, StructuredClientBuilder,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use tracing_subscriber::{fmt, EnvFilter};

/// Configuration for how an account should be forecast.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ForecastConfig {
    /// The forecast method to use
    #[schemars(
        description = "Method: 'growth_rate', 'fixed', 'percentage_of_revenue', or 'manual'"
    )]
    pub method: String,
    /// Growth rate as decimal (e.g., 0.05 for 5% growth). Used with 'growth_rate' method.
    pub growth_rate: Option<f64>,
    /// Fixed value to use. Used with 'fixed' method.
    pub fixed_value: Option<f64>,
    /// Percentage of revenue (e.g., 0.30 for 30%). Used with 'percentage_of_revenue' method.
    pub revenue_percentage: Option<f64>,
}

/// A single account in the financial statement.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Account {
    /// Unique account code (e.g., "REV001", "EXP001")
    pub code: String,
    /// Human-readable account name
    pub name: String,
    /// Account category: "revenue", "expense", "asset", "liability", "equity"
    #[schemars(description = "One of: 'revenue', 'expense', 'asset', 'liability', 'equity'")]
    pub category: String,
    /// Historical values by period (e.g., {"2023-Q1": 100000, "2023-Q2": 105000})
    ///
    /// The adapter treats the map as a list of `{key, value}` pairs for the LLM,
    /// then deserializes it back into a `HashMap`.
    #[serde(with = "gemini_structured_output::adapter::map")]
    #[schemars(with = "Vec<KeyValue<String, f64>>")]
    pub historical: HashMap<String, f64>,
    /// Forecast values by period
    #[serde(with = "gemini_structured_output::adapter::map")]
    #[schemars(with = "Vec<KeyValue<String, f64>>")]
    pub forecast: HashMap<String, f64>,
    /// Configuration for forecasting this account
    pub forecast_config: ForecastConfig,
}

/// Complete financial model with accounts and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FinancialModel {
    /// Company name
    pub company_name: String,
    /// Currency code (e.g., "USD", "NZD")
    #[schemars(description = "3-letter ISO currency code")]
    pub currency: String,
    /// All accounts in the model
    pub accounts: Vec<Account>,
    /// Periods included in the model
    pub periods: Vec<String>,
    /// Summary metrics
    pub summary: FinancialSummary,
}

/// Summary financial metrics.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FinancialSummary {
    /// Total revenue for the most recent period
    #[serde(default)]
    pub total_revenue: Option<f64>,
    /// Total expenses for the most recent period
    #[serde(default)]
    pub total_expenses: Option<f64>,
    /// Net income (revenue - expenses)
    #[serde(default)]
    pub net_income: Option<f64>,
    /// Gross margin percentage
    #[serde(default)]
    pub gross_margin_percent: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,gemini_structured_output=debug"));
    fmt().with_env_filter(filter).init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model = Model::Custom("models/gemini-2.5-flash-preview-09-2025".to_string());

    println!("=== Generated JSON Schema ===");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&FinancialModel::gemini_schema())?
    );

    // Build client with ReplaceWhole array strategy (safest for complex arrays)
    let client = StructuredClientBuilder::new(api_key)
        .with_model(model)
        .with_array_strategy(ArrayPatchStrategy::ReplaceWhole)
        .with_refinement_retries(3)
        .build()?;

    // Create initial financial statement with historical data
    let initial_model = create_sample_financial_model();

    println!("=== Initial Financial Model ===");
    print_model_summary(&initial_model);

    let system_instruction = "\
You are a financial analyst. Given the financial data, create a complete financial model \
with appropriate forecast configurations for each account. Use realistic growth rates \
based on the historical trends. Ensure all calculations are accurate. \
IMPORTANT: Forecast only the next 4 quarters (2024-Q1 to 2024-Q4). \
Do not generate data beyond these periods.";

    let user_text = format!(
        "Create a financial model for {} with these accounts and historical data:\n{}",
        initial_model.company_name,
        serde_json::to_string_pretty(&initial_model)?
    );

    println!("\n=== Generating AI-enhanced model... ===");

    let build_request = |user_text: String| {
        client
            .request::<FinancialModel>()
            .system(system_instruction)
            .user_text(user_text)
            .with_generation_config(GenerationConfig {
                max_output_tokens: Some(10_000),
                temperature: Some(0.2),
                ..Default::default()
            })
            .retries(3)
            .max_parse_attempts(3)
    };

    let enhanced_model: FinancialModel = if std::env::var("STREAM_FORECAST").is_ok() {
        println!("Streaming raw model output...");
        let mut stream = build_request(user_text.clone()).stream().await?;
        let mut final_outcome = None;
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Chunk(chunk) => {
                    print!("{chunk}");
                }
                StreamEvent::Complete(outcome) => {
                    final_outcome = Some(outcome);
                }
            }
        }
        println!();
        final_outcome
            .expect("stream should produce a complete event")
            .value
    } else {
        build_request(user_text).execute().await?.value
    };

    println!("\n=== AI-Enhanced Model ===");
    print_model_summary(&enhanced_model);

    println!("\n=== Generating Q1 2025 Forecast via Refinement ===");

    let forecast_instruction = r#"
Generate forecast values for Q1 2025 (period "2025-Q1") for all accounts based on their forecast_config:
1. For accounts with method="growth_rate": Apply the growth_rate to the last historical/forecast value
2. For accounts with method="percentage_of_revenue": Calculate as percentage of the forecasted revenue
3. For accounts with method="fixed": Use the fixed_value
4. Add "2025-Q1" to the periods array
5. Update the summary with the new forecasted values. Do not return null; use numeric values.
"#;

    let forecast_result = client.refine(&enhanced_model, forecast_instruction).await?;

    println!("\n=== Forecasted Model ===");
    print_model_summary(&forecast_result.value);

    if let Some(account) = forecast_result.value.accounts.first() {
        match account.forecast.get("2025-Q1") {
            Some(val) => println!(
                "\n✅ Retrieved '2025-Q1' from HashMap for {}: ${:.2}",
                account.code, val
            ),
            None => println!(
                "\n❌ Missing '2025-Q1' entry in forecast map for {}",
                account.code
            ),
        }
    }

    println!("\n=== Refinement Statistics ===");
    println!("Total attempts: {}", forecast_result.attempts.len());
    for (i, attempt) in forecast_result.attempts.iter().enumerate() {
        println!(
            "  Attempt {}: success={}, error={:?}",
            i + 1,
            attempt.success,
            attempt.error.as_ref().map(|e| truncate(e, 100))
        );
    }

    // Verify the forecast was applied
    println!("\n=== Forecast Verification ===");
    for account in &forecast_result.value.accounts {
        if let Some(q1_2025) = account.forecast.get("2025-Q1") {
            println!(
                "  {} ({}): ${:.2} (method: {})",
                account.name, account.code, q1_2025, account.forecast_config.method
            );
        }
    }

    Ok(())
}

fn create_sample_financial_model() -> FinancialModel {
    let mut accounts = Vec::new();

    // Revenue account
    let mut rev_historical = HashMap::new();
    rev_historical.insert("2023-Q1".to_string(), 100000.0);
    rev_historical.insert("2023-Q2".to_string(), 110000.0);
    rev_historical.insert("2023-Q3".to_string(), 115000.0);
    rev_historical.insert("2023-Q4".to_string(), 125000.0);

    accounts.push(Account {
        code: "REV001".to_string(),
        name: "Product Revenue".to_string(),
        category: "revenue".to_string(),
        historical: rev_historical,
        forecast: HashMap::new(),
        forecast_config: ForecastConfig {
            method: "growth_rate".to_string(),
            growth_rate: Some(0.08),
            fixed_value: None,
            revenue_percentage: None,
        },
    });

    // Service revenue
    let mut svc_historical = HashMap::new();
    svc_historical.insert("2023-Q1".to_string(), 25000.0);
    svc_historical.insert("2023-Q2".to_string(), 27000.0);
    svc_historical.insert("2023-Q3".to_string(), 28000.0);
    svc_historical.insert("2023-Q4".to_string(), 30000.0);

    accounts.push(Account {
        code: "REV002".to_string(),
        name: "Service Revenue".to_string(),
        category: "revenue".to_string(),
        historical: svc_historical,
        forecast: HashMap::new(),
        forecast_config: ForecastConfig {
            method: "growth_rate".to_string(),
            growth_rate: Some(0.05),
            fixed_value: None,
            revenue_percentage: None,
        },
    });

    // Cost of goods sold
    let mut cogs_historical = HashMap::new();
    cogs_historical.insert("2023-Q1".to_string(), 40000.0);
    cogs_historical.insert("2023-Q2".to_string(), 44000.0);
    cogs_historical.insert("2023-Q3".to_string(), 46000.0);
    cogs_historical.insert("2023-Q4".to_string(), 50000.0);

    accounts.push(Account {
        code: "EXP001".to_string(),
        name: "Cost of Goods Sold".to_string(),
        category: "expense".to_string(),
        historical: cogs_historical,
        forecast: HashMap::new(),
        forecast_config: ForecastConfig {
            method: "percentage_of_revenue".to_string(),
            growth_rate: None,
            fixed_value: None,
            revenue_percentage: Some(0.35),
        },
    });

    // Operating expenses
    let mut opex_historical = HashMap::new();
    opex_historical.insert("2023-Q1".to_string(), 35000.0);
    opex_historical.insert("2023-Q2".to_string(), 36000.0);
    opex_historical.insert("2023-Q3".to_string(), 37000.0);
    opex_historical.insert("2023-Q4".to_string(), 38000.0);

    accounts.push(Account {
        code: "EXP002".to_string(),
        name: "Operating Expenses".to_string(),
        category: "expense".to_string(),
        historical: opex_historical,
        forecast: HashMap::new(),
        forecast_config: ForecastConfig {
            method: "growth_rate".to_string(),
            growth_rate: Some(0.02),
            fixed_value: None,
            revenue_percentage: None,
        },
    });

    // Rent (fixed)
    let mut rent_historical = HashMap::new();
    rent_historical.insert("2023-Q1".to_string(), 15000.0);
    rent_historical.insert("2023-Q2".to_string(), 15000.0);
    rent_historical.insert("2023-Q3".to_string(), 15000.0);
    rent_historical.insert("2023-Q4".to_string(), 15000.0);

    accounts.push(Account {
        code: "EXP003".to_string(),
        name: "Rent".to_string(),
        category: "expense".to_string(),
        historical: rent_historical,
        forecast: HashMap::new(),
        forecast_config: ForecastConfig {
            method: "fixed".to_string(),
            growth_rate: None,
            fixed_value: Some(15000.0),
            revenue_percentage: None,
        },
    });

    // Calculate summary for Q4 2023
    let total_revenue = 125000.0 + 30000.0;
    let total_expenses = 50000.0 + 38000.0 + 15000.0;

    FinancialModel {
        company_name: "TechCorp Inc.".to_string(),
        currency: "USD".to_string(),
        accounts,
        periods: vec![
            "2023-Q1".to_string(),
            "2023-Q2".to_string(),
            "2023-Q3".to_string(),
            "2023-Q4".to_string(),
        ],
        summary: FinancialSummary {
            total_revenue: Some(total_revenue),
            total_expenses: Some(total_expenses),
            net_income: Some(total_revenue - total_expenses),
            gross_margin_percent: Some(((total_revenue - 50000.0) / total_revenue) * 100.0),
        },
    }
}

fn print_model_summary(model: &FinancialModel) {
    println!("Company: {}", model.company_name);
    println!("Currency: {}", model.currency);
    println!("Periods: {:?}", model.periods);
    println!("Accounts: {}", model.accounts.len());

    for account in &model.accounts {
        println!(
            "  - {} ({}): {} | forecast_method={}",
            account.name, account.code, account.category, account.forecast_config.method
        );

        // Show latest historical
        if let Some(latest) = model.periods.last() {
            if let Some(val) = account.historical.get(latest) {
                println!("      Historical {}: ${:.2}", latest, val);
            }
        }

        // Show forecast if any
        if !account.forecast.is_empty() {
            for (period, value) in &account.forecast {
                println!("      Forecast {}: ${:.2}", period, value);
            }
        }
    }

    println!("\nSummary (latest period):");
    println!(
        "  Total Revenue: ${:.2}",
        model.summary.total_revenue.unwrap_or(0.0)
    );
    println!(
        "  Total Expenses: ${:.2}",
        model.summary.total_expenses.unwrap_or(0.0)
    );
    println!(
        "  Net Income: ${:.2}",
        model.summary.net_income.unwrap_or(0.0)
    );
    println!(
        "  Gross Margin: {:.1}%",
        model.summary.gross_margin_percent.unwrap_or(0.0)
    );
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
