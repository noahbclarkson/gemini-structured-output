use gemini_rust::Model;
use gemini_structured_output::StructuredClientBuilder;
use std::env;
use xero_forecasting::config::{FullForecastConfig, PnlProcessor, ForecastModel};

#[tokio::test]
async fn test_real_xero_forecast_config_with_gemini() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini3Flash)
        .with_default_retries(2)
        .with_default_parse_attempts(2)
        .build()
        .expect("Failed to build client");

    let prompt = r#"
Generate a comprehensive financial forecast configuration for a software company with the following requirements:

1. Configure the P&L account "Revenue - Subscriptions" with an Auto forecasting model
2. Configure the P&L account "Operating Expenses - Salaries" with a Linear Regression model
3. Configure the P&L account "Marketing Expenses" with a fixed value model (value: 5000)
4. Add a Balance Sheet configuration with at least one account override

Return a complete FullForecastConfig matching the schema.
"#;

    println!("Sending request to Gemini...");

    let result = client
        .request::<FullForecastConfig>()
        .user_text(prompt)
        .temperature(0.7)
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            let config = outcome.value;
            println!("\n✅ Successfully generated forecast config!");
            println!("\nGenerated Config:");
            println!("{}", serde_json::to_string_pretty(&config).unwrap());

            assert!(config.pnl_config.forecast_config.account_overrides.len() >= 3,
                "Expected at least 3 P&L account overrides");

            let has_auto_model = config.pnl_config.forecast_config.account_overrides.values().any(|o| {
                matches!(
                    o.processor,
                    Some(PnlProcessor::Model(ForecastModel::Auto))
                )
            });
            assert!(has_auto_model, "Expected at least one Auto forecast model");

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Test failed: {}", e);
        }
    }
}
