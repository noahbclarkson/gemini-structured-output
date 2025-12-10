use gemini_rust::Model;
use gemini_structured_output::{CachePolicy, CacheSettings, StructuredClientBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{env, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ProductBrief {
    /// Human-friendly product name (short, brand-aware).
    name: String,
    /// A richly worded, marketing-ready description emphasizing differentiators, target users, and benefits.
    description: String,
    /// Price in USD; include cents.
    price: f32,
    /// Key selling points; ordered by importance.
    highlights: Vec<String>,
    /// Technical specs and compliance notes.
    specs: Specs,
    /// Target persona summary.
    persona: Persona,
    /// Compliance and safety statements for regulated industries.
    compliance: Compliance,
    /// Packaging and shipping details.
    packaging: Packaging,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Specs {
    /// Supported operating systems.
    os_support: Vec<String>,
    /// Connectivity standards (e.g., Wi-Fi 6, BT 5.3).
    connectivity: Vec<String>,
    /// Battery life estimates in hours (idle/active).
    battery_life_hours: BatteryLife,
    /// Physical dimensions in millimeters (LxWxH).
    dimensions_mm: Dimensions,
    /// Total weight in grams.
    weight_grams: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BatteryLife {
    /// Idle runtime in hours.
    idle: f32,
    /// Active runtime in hours under typical workload.
    active: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Dimensions {
    length_mm: f32,
    width_mm: f32,
    height_mm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Persona {
    /// Primary buyer role (e.g., IT Manager, Operations Lead).
    role: String,
    /// Company size band (e.g., SMB, Mid-Market, Enterprise).
    company_size: String,
    /// Top 3 pain points this product solves.
    pain_points: Vec<String>,
    /// Success metric the buyer cares about most.
    success_metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Compliance {
    /// List of applicable standards (e.g., ISO 27001, SOC2, HIPAA).
    standards: Vec<String>,
    /// Region-specific notes (e.g., GDPR, CCPA).
    regional: Vec<String>,
    /// Safety disclaimers and handling warnings.
    safety: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Packaging {
    /// Materials used (e.g., recycled cardboard, plastic-free).
    materials: Vec<String>,
    /// Unboxing experience highlights.
    unboxing_notes: Vec<String>,
    /// Shipping dimensions (LxWxH) in millimeters.
    ship_dimensions_mm: Dimensions,
    /// Shipping weight in grams.
    ship_weight_grams: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run examples");
    let model = Model::Custom("models/gemini-2.5-flash-preview-09-2025".to_string());

    // Enable caching with a 10 minute TTL to reuse the schema/system prompt.
    let client = StructuredClientBuilder::new(api_key)
        .with_model(model.clone())
        .with_cache_policy(CachePolicy::Enabled {
            ttl: Duration::from_secs(600),
        })
        .build()?;

    // Use a stable cache key for this schema/system combo.
    let cache_settings = CacheSettings::with_key("product-brief-cache");

    // First call: builds cache
    let first = client
        .request::<ProductBrief>()
        .system("Return a richly detailed product brief matching the schema.")
        .user_text(
            "Product: SuperWidget Pro; Description: Ruggedized IoT gateway with \
             Wi-Fi 6, BT 5.3, and 48h battery; Price: 249.99",
        )
        .with_cache(cache_settings.clone())
        .execute()
        .await?
        .value;

    // Second call: should reuse cached content (lower prompt tokens from the API)
    let second = client
        .request::<ProductBrief>()
        .system("Return a richly detailed product brief matching the schema.")
        .user_text(
            "Product: SuperWidget Ultra; Description: Edge AI box with TPU and dual 10GbE; Price: 499.00",
        )
        .with_cache(cache_settings)
        .execute()
        .await?
        .value;

    println!("Model: {}", model);
    println!("First result: {first:#?}");
    println!("Second result: {second:#?}");

    Ok(())
}
