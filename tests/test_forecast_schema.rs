use gemini_structured_output::schema::GeminiStructured;
use xero_forecasting::config::pnl::PnlProcessor;
use xero_forecasting::config::ForecastModel;

#[test]
fn inspect_pnl_processor_schema() {
    let schema = PnlProcessor::gemini_schema();
    println!("\n=== PnlProcessor Schema ===");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}

#[test]
fn inspect_forecast_model_schema() {
    let schema = ForecastModel::gemini_schema();
    println!("\n=== ForecastModel Schema ===");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}

#[test]
fn test_manual_pnl_processor_deser() {
    // What Gemini outputs
    let gemini_output = serde_json::json!({
        "model": "mstl",
        "seasonalPeriods": [12],
        "trendModel": "ets"
    });

    println!("\nGemini output:");
    println!("{}", serde_json::to_string_pretty(&gemini_output).unwrap());

    // What serde expects
    let correct_format = serde_json::json!({
        "model": {
            "Mstl": {
                "seasonalPeriods": [12],
                "trendModel": "Ets"
            }
        }
    });

    println!("\nCorrect format:");
    println!("{}", serde_json::to_string_pretty(&correct_format).unwrap());

    let result = serde_json::from_value::<PnlProcessor>(correct_format.clone());
    match result {
        Ok(processor) => println!("\n✅ Correct format deserializes: {:?}", processor),
        Err(e) => println!("\n❌ Error: {}", e),
    }

    let result2 = serde_json::from_value::<PnlProcessor>(gemini_output.clone());
    match result2 {
        Ok(processor) => println!("\n✅ Gemini format deserializes: {:?}", processor),
        Err(e) => println!("\n❌ Gemini format error (expected): {}", e),
    }
}
