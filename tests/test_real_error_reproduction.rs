use gemini_structured_output::schema::{
    prune_null_fields, unflatten_externally_tagged_enums, GeminiStructured,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ForecastConfig {
    periods_to_forecast: u32,
    default_model: ModelType,
    fallback_model: ModelType,
    account_overrides: HashMap<String, AccountOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct AccountOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    is_non_cash: Option<bool>,
    processor: Processor,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ModelType {
    Auto,
    LastValue,
}

// This is an externally-tagged enum (default serde representation)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "PascalCase")]
enum Processor {
    Mstl {
        seasonal_periods: Vec<u32>,
        trend_model: String,
    },
    Tax {
        tax_rate: f64,
        allow_negative_tax: bool,
    },
    LastValue,
}

#[test]
fn test_real_error_reproduction() {
    // This is the actual response from Gemini - FLATTENED format with all variant fields merged
    // and null values for fields that don't belong to the active variant
    let mut response = json!({
        "periodsToForecast": 12,
        "defaultModel": {"type": "auto"},
        "fallbackModel": {"type": "lastValue"},
        "accountOverrides": {
            "Access Advisors - Sales": {
                "processor": {
                    // Flattened Mstl variant - has null fields from other variants
                    "seasonal_periods": [12],
                    "trend_model": "ets",
                    "tax_rate": null,
                    "allow_negative_tax": null
                }
            },
            "Tax Provision": {
                "processor": {
                    // Flattened Tax variant
                    "seasonal_periods": null,
                    "trend_model": null,
                    "tax_rate": 0.28,
                    "allow_negative_tax": false
                }
            },
            "Depreciation": {
                "isNonCash": true,
                "processor": {
                    // Flattened LastValue variant (unit variant)
                    "seasonal_periods": null,
                    "trend_model": null,
                    "tax_rate": null,
                    "allow_negative_tax": null
                }
            }
        }
    });

    println!("Response before processing:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let schema = ForecastConfig::gemini_schema();
    println!("\nSchema:");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    // Apply the same transformations as the real code
    prune_null_fields(&mut response);
    unflatten_externally_tagged_enums(&mut response, &schema);

    println!("\nResponse after unflatten_externally_tagged_enums:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Try to deserialize
    let result: Result<ForecastConfig, _> = serde_json::from_value(response.clone());

    match &result {
        Ok(config) => {
            println!("\n✅ Successfully deserialized!");
            println!("Config: {:?}", config);
        }
        Err(e) => {
            println!("\n❌ Deserialization failed: {}", e);
            println!("Response that failed:");
            println!("{}", serde_json::to_string_pretty(&response).unwrap());
        }
    }

    assert!(result.is_ok(), "Should deserialize successfully");
}
