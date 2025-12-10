//! Example demonstrating the procedural macros for tools and validation.
//!
//! This example shows how to use:
//! - `#[gemini_tool]` attribute macro for defining tool functions
//! - `#[derive(GeminiValidated)]` for declarative validation rules
//! - `FallbackStrategy` for model escalation
//!
//! Run with: `cargo run --example macro_tools --features macros`

use gemini_structured_output::prelude::*;
use gemini_structured_output::tools::ToolError;
use gemini_structured_output::{gemini_tool, GeminiValidated};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// === Tool Input/Output Types ===

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct StockRequest {
    /// Stock ticker symbol (e.g., "AAPL", "GOOGL")
    symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct StockPrice {
    /// Stock ticker symbol
    symbol: String,
    /// Current price in USD
    price: f64,
    /// Price change percentage
    change_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherRequest {
    /// City name
    city: String,
    /// Country code (e.g., "US", "UK")
    country: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherInfo {
    /// City name
    city: String,
    /// Temperature in Celsius
    temperature: f64,
    /// Weather condition description
    condition: String,
    /// Humidity percentage
    humidity: i32,
}

// === Tool Definitions using #[gemini_tool] ===

/// Look up the current price of a stock.
#[gemini_tool(description = "Look up the current price of a stock by its ticker symbol")]
async fn get_stock_price(args: StockRequest) -> std::result::Result<StockPrice, ToolError> {
    // Simulated stock price lookup
    let price = match args.symbol.to_uppercase().as_str() {
        "AAPL" => 178.50,
        "GOOGL" => 141.25,
        "MSFT" => 378.90,
        "AMZN" => 178.75,
        _ => 100.00, // Default price for unknown stocks
    };

    Ok(StockPrice {
        symbol: args.symbol.to_uppercase(),
        price,
        change_percent: 1.25, // Simulated change
    })
}

/// Get current weather information for a city.
#[gemini_tool(
    name = "weather_lookup",
    description = "Get current weather information for a city"
)]
async fn get_weather(args: WeatherRequest) -> std::result::Result<WeatherInfo, ToolError> {
    // Simulated weather lookup
    Ok(WeatherInfo {
        city: args.city.clone(),
        temperature: 22.5,
        condition: "Partly Cloudy".to_string(),
        humidity: 65,
    })
}

// === Validated Struct using #[derive(GeminiValidated)] ===

/// Custom validator function for email format
fn validate_email(email: &str) -> Option<String> {
    if !email.contains('@') || !email.contains('.') {
        Some("invalid email format".to_string())
    } else {
        None
    }
}

/// A user profile with declarative validation rules.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, GeminiValidated)]
struct UserProfile {
    /// User's full name (required, max 100 chars)
    #[gemini(non_empty, max_len = 100)]
    name: String,

    /// User's email address
    #[gemini(validate_with = "validate_email", error_message = "Email validation")]
    email: String,

    /// User's age (must be between 0 and 150)
    #[gemini(min = 0.0, max = 150.0)]
    age: i32,

    /// User bio (optional, max 500 chars)
    #[gemini(max_len = 500)]
    bio: String,

    /// User's tags/interests (at least 1 required)
    #[gemini(min_len = 1)]
    interests: Vec<String>,
}

// === Output Type for Structured Generation ===

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MarketSummary {
    stocks: Vec<StockPrice>,
    analysis: String,
    recommendation: String,
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("=== Gemini Structured Output - Macro Tools Example ===\n");

    // Get API key from environment
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // === Example 1: Using macro-defined tools ===
    println!("1. Setting up client with macro-defined tools...\n");

    // Build client with model escalation enabled
    let client = StructuredClientBuilder::new(&api_key)
        .with_model(Model::Custom(
            "models/gemini-2.5-flash-preview-09-2025".to_string(),
        ))
        .with_fallback_strategy(FallbackStrategy::Escalate {
            after_attempts: 2,
            target: Model::Gemini25Pro,
        })
        .build()?;

    // Register tools using the macro-generated registrars
    let tools = ToolRegistry::new()
        .register_tool(get_stock_price_tool::registrar())
        .register_tool(get_weather_tool::registrar());

    println!("   Registered tools:");
    println!(
        "   - {}: {}",
        get_stock_price_tool::NAME,
        get_stock_price_tool::DESCRIPTION
    );
    println!(
        "   - {}: {}",
        get_weather_tool::NAME,
        get_weather_tool::DESCRIPTION
    );

    // === Example 2: Testing validation ===
    println!("\n2. Testing GeminiValidated derive macro...\n");

    // Valid profile
    let valid_profile = UserProfile {
        name: "John Doe".to_string(),
        email: "john@example.com".to_string(),
        age: 30,
        bio: "Software developer interested in AI".to_string(),
        interests: vec!["coding".to_string(), "AI".to_string()],
    };

    match valid_profile.gemini_validate() {
        Some(err) => println!("   Valid profile failed: {}", err),
        None => println!("   Valid profile passed validation"),
    }

    // Invalid profile (empty name)
    let invalid_name = UserProfile {
        name: "".to_string(),
        email: "john@example.com".to_string(),
        age: 30,
        bio: "Test".to_string(),
        interests: vec!["test".to_string()],
    };

    match invalid_name.gemini_validate() {
        Some(err) => println!("   Empty name detected: {}", err),
        None => println!("   Validation should have failed!"),
    }

    // Invalid profile (bad email)
    let invalid_email = UserProfile {
        name: "Jane Doe".to_string(),
        email: "invalid-email".to_string(),
        age: 25,
        bio: "Test".to_string(),
        interests: vec!["test".to_string()],
    };

    match invalid_email.gemini_validate() {
        Some(err) => println!("   Invalid email detected: {}", err),
        None => println!("   Validation should have failed!"),
    }

    // Invalid profile (age out of range)
    let invalid_age = UserProfile {
        name: "Bob Smith".to_string(),
        email: "bob@test.com".to_string(),
        age: 200,
        bio: "Test".to_string(),
        interests: vec!["test".to_string()],
    };

    match invalid_age.gemini_validate() {
        Some(err) => println!("   Invalid age detected: {}", err),
        None => println!("   Validation should have failed!"),
    }

    // === Example 3: Using tools with structured generation ===
    println!("\n3. Generating structured output with tools...\n");

    let result: GenerationOutcome<MarketSummary> = client
        .request::<MarketSummary>()
        .system("You are a financial analyst. Use the available tools to gather data.")
        .user_text("Get the stock prices for AAPL and GOOGL, then provide a brief market summary.")
        .with_tools(tools)
        .max_tool_steps(5)
        .execute()
        .await?;

    println!("   Market Summary:");
    println!("   Stocks analyzed: {:?}", result.value.stocks.len());
    println!("   Analysis: {}", result.value.analysis);
    println!("   Recommendation: {}", result.value.recommendation);

    if let Some(usage) = result.usage {
        println!("\n   Token usage:");
        println!("   - Prompt tokens: {:?}", usage.prompt_token_count);
        println!("   - Response tokens: {:?}", usage.candidates_token_count);
    }

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
