use gemini_structured_output::schema::GeminiStructured;
use serde_json::json;
use xero_forecasting::config::pnl::{PnlProcessor, PnlAccountOverride};

#[test]
fn test_pnl_account_override_unflatten() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("trace")
        .with_test_writer()
        .try_init();

    // Gemini's flattened output for a PnlAccountOverride with nested enum
    let mut gemini_output = json!({
        "processor": {
            "model": "mstl",
            "seasonalPeriods": [12],
            "trendModel": "ets"
        },
        "isNonCash": false,
        "constraints": { "nonNegative": true }
    });

    println!("\n=== Gemini Output (Flattened) ===");
    println!("{}", serde_json::to_string_pretty(&gemini_output).unwrap());

    let schema = PnlAccountOverride::gemini_schema();

    println!("\n=== PnlAccountOverride Schema ===");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap().lines().take(100).collect::<Vec<_>>().join("\n"));

    println!("\n=== Processor Field Schema ===");
    if let Some(processor_schema) = schema.get("properties").and_then(|p| p.get("processor")) {
        println!("{}", serde_json::to_string_pretty(&processor_schema).unwrap().lines().take(50).collect::<Vec<_>>().join("\n"));

        // Check if x-anyOf-original exists
        if let Some(x_any_of) = processor_schema.get("x-anyOf-original") {
            println!("\n=== x-anyOf-original exists! ===");
            println!("Number of variants: {}", x_any_of.as_array().map(|a| a.len()).unwrap_or(0));

            // Check if the first variant has anyOf
            if let Some(first_variant) = x_any_of.as_array().and_then(|a| a.get(0)) {
                if first_variant.get("anyOf").is_some() {
                    println!("First variant has anyOf - this is the actual PnlProcessor enum!");
                }
            }
        }
    }

    gemini_structured_output::schema::unflatten_externally_tagged_enums(&mut gemini_output, &schema);

    println!("\n=== After Unflatten ===");
    println!("{}", serde_json::to_string_pretty(&gemini_output).unwrap());

    // Try to deserialize
    let result = serde_json::from_value::<PnlAccountOverride>(gemini_output);
    match result {
        Ok(override_config) => println!("\n✅ Success! Deserialized to: {:?}", override_config),
        Err(e) => {
            eprintln!("\n❌ Failed to deserialize: {}", e);
            panic!("Deserialization failed");
        }
    }
}

#[test]
fn test_pnl_processor_unflatten() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("trace")
        .with_test_writer()
        .try_init();

    // Gemini's flattened output for a nested enum
    let mut gemini_output = json!({
        "model": "mstl",
        "seasonalPeriods": [12],
        "trendModel": "ets"
    });

    println!("\n=== Gemini Output (Flattened) ===");
    println!("{}", serde_json::to_string_pretty(&gemini_output).unwrap());

    let schema = PnlProcessor::gemini_schema();

    println!("\n=== Schema (anyOf variants) ===");
    if let Some(any_of) = schema.get("anyOf") {
        if let Some(variants) = any_of.as_array() {
            for (i, variant) in variants.iter().enumerate() {
                println!("\nVariant {}: {}", i, serde_json::to_string_pretty(variant).unwrap().lines().take(20).collect::<Vec<_>>().join("\n"));
            }
        }
    }

    gemini_structured_output::schema::unflatten_externally_tagged_enums(&mut gemini_output, &schema);

    println!("\n=== After Unflatten ===");
    println!("{}", serde_json::to_string_pretty(&gemini_output).unwrap());

    // Try to deserialize
    let result = serde_json::from_value::<PnlProcessor>(gemini_output);
    match result {
        Ok(processor) => println!("\n✅ Success! Deserialized to: {:?}", processor),
        Err(e) => {
            eprintln!("\n❌ Failed to deserialize: {}", e);
            panic!("Deserialization failed");
        }
    }
}

#[test]
fn test_simple_flattened_detection() {
    use gemini_structured_output::schema::GeminiStructured;

    // Get the schema for PnlProcessor
    let schema = PnlProcessor::gemini_schema();

    println!("\n=== PnlProcessor Schema (first 100 lines) ===");
    let schema_str = serde_json::to_string_pretty(&schema).unwrap();
    for (i, line) in schema_str.lines().enumerate().take(100) {
        println!("{}: {}", i + 1, line);
    }

    // Check if it has anyOf
    if let Some(any_of) = schema.get("anyOf") {
        println!("\n✅ Schema has anyOf with {} variants", any_of.as_array().map(|a| a.len()).unwrap_or(0));
    } else {
        println!("\n❌ Schema does NOT have anyOf");
    }
}
