use gemini_structured_output::schema::{normalize_json_response, prune_null_fields, unflatten_externally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct TestData {
    processor: Processor,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "PascalCase")]
enum Processor {
    Model {
        model: String,
        seasonal_periods: Vec<u32>,
        trend_model: String,
    },
    Calculation {
        calculation: Vec<serde_json::Value>,
    },
    TaxCalculation {
        tax_calculation: Vec<serde_json::Value>,
    },
}

#[test]
fn test_processor_with_model_discriminator() {
    // Gemini's flattened response
    let mut response = json!({
        "processor": {
            "model": "mstl",
            "seasonal_periods": [12],
            "trend_model": "ets",
            // These would be null in reality, but pruned
        }
    });

    println!("Before processing:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let schema = TestData::gemini_schema();

    prune_null_fields(&mut response);
    unflatten_externally_tagged_enums(&mut response, &schema);

    println!("\nAfter unflatten:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let result: Result<TestData, _> = serde_json::from_value(response.clone());
    match &result {
        Ok(data) => println!("\n✅ Success: {:?}", data),
        Err(e) => println!("\n❌ Failed: {}", e),
    }

    assert!(result.is_ok(), "Should deserialize");
}

#[test]
fn test_processor_with_calculation_field() {
    let mut response = json!({
        "processor": {
            "calculation": ["sumOfAccounts", "Account1", "multiply", 0.35]
        }
    });

    println!("Before processing:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let schema = TestData::gemini_schema();

    prune_null_fields(&mut response);
    unflatten_externally_tagged_enums(&mut response, &schema);

    println!("\nAfter unflatten:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let result: Result<TestData, _> = serde_json::from_value(response.clone());
    match &result {
        Ok(data) => println!("\n✅ Success: {:?}", data),
        Err(e) => println!("\n❌ Failed: {}", e),
    }

    assert!(result.is_ok(), "Should deserialize");
}
