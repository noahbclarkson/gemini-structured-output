use gemini_rust::Model;
use gemini_structured_output::{StructuredClientBuilder, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherRequest {
    city: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    summary: String,
    temperature_c: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherReport {
    location: String,
    advice: String,
    temperature: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run examples");
    let model = Model::Custom("models/gemini-2.5-flash-preview-09-2025".to_string());
    let client = StructuredClientBuilder::new(api_key)
        .with_model(model)
        .build()?;

    // Register tool with implementation
    let tools = ToolRegistry::new().register_with_handler::<WeatherRequest, WeatherResult, _, _>(
        "get_weather",
        "Return current weather for a city",
        |req| async move {
            println!("> Tool executed for: {}", req.city);
            Ok(WeatherResult {
                summary: format!("Sunny in {}", req.city),
                temperature_c: 18.5,
            })
        },
    );

    let outcome = client
        .request::<WeatherReport>()
        .system("Use tools to get weather then advise attire.")
        .user_text("What should I wear in Wellington today?")
        .with_tools(tools)
        .execute()
        .await?;

    println!("Final Report: {:#?}", outcome.value);
    Ok(())
}
