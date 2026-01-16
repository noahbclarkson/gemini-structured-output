use gemini_structured_output::schema::GeminiStructured;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use xero_forecasting::config::FullForecastConfig;

#[test]
fn debug_account_overrides_schema() {
    // First check the RAW schema before Gemini transformations
    let settings = SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let generator = SchemaGenerator::new(settings);
    let raw_schema = generator.into_root_schema_for::<FullForecastConfig>();
    let raw_json = serde_json::to_value(&raw_schema).unwrap();

    println!("\n=== Checking RAW schema (before Gemini transformations) ===");
    if let Some(raw_ao) = raw_json
        .get("properties")
        .and_then(|p| p.get("pnlConfig"))
        .and_then(|pc| pc.get("properties"))
        .and_then(|p| p.get("forecastConfig"))
        .and_then(|fc| fc.get("properties"))
        .and_then(|p| p.get("accountOverrides"))
    {
        println!("Raw accountOverrides type: {:?}", raw_ao.get("type"));
        println!("Raw accountOverrides has properties: {}", raw_ao.get("properties").is_some());
        println!("Raw accountOverrides has additionalProperties: {}", raw_ao.get("additionalProperties").is_some());
        if let Some(add_props) = raw_ao.get("additionalProperties") {
            println!("\n=== Raw additionalProperties (first 30 lines) ===");
            println!("{}", serde_json::to_string_pretty(add_props).unwrap().lines().take(30).collect::<Vec<_>>().join("\n"));
        }
    }

    let schema = FullForecastConfig::gemini_schema();

    println!("\n=== Full Schema (first 100 lines) ===");
    let schema_str = serde_json::to_string_pretty(&schema).unwrap();
    for (i, line) in schema_str.lines().take(100).enumerate() {
        println!("{}: {}", i + 1, line);
    }

    // Navigate to accountOverrides
    let pnl_config = schema
        .get("properties")
        .and_then(|p| p.get("pnlConfig"));

    println!("\n=== pnlConfig found: {} ===", pnl_config.is_some());

    if let Some(pnl) = pnl_config {
        let forecast_config = pnl
            .get("properties")
            .and_then(|p| p.get("forecastConfig"));

        println!("=== forecastConfig found: {} ===", forecast_config.is_some());

        if let Some(fc) = forecast_config {
            let account_overrides = fc
                .get("properties")
                .and_then(|p| p.get("accountOverrides"));

            println!("=== accountOverrides found: {} ===", account_overrides.is_some());

            if let Some(ao) = account_overrides {
                println!("\n=== accountOverrides type: {:?} ===", ao.get("type"));
                println!("=== accountOverrides has properties: {} ===", ao.get("properties").is_some());
                println!("=== accountOverrides has additionalProperties: {} ===", ao.get("additionalProperties").is_some());

                println!("\n=== accountOverrides schema ===");
                println!("{}", serde_json::to_string_pretty(ao).unwrap());

                println!("\n=== Checking for x-additionalProperties-original ===");
                println!("Has x-additionalProperties-original: {}", ao.get("x-additionalProperties-original").is_some());

                if let Some(x_add_props) = ao.get("x-additionalProperties-original") {
                    println!("\n=== x-additionalProperties-original ===");
                    println!("{}", serde_json::to_string_pretty(x_add_props).unwrap().lines().take(50).collect::<Vec<_>>().join("\n"));
                }
            }
        }
    }
}
