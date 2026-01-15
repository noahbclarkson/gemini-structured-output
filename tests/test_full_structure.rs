use gemini_structured_output::schema::{normalize_json_response, prune_null_fields, unflatten_externally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Config {
    pnl_config: PnlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct PnlConfig {
    forecast_config: ForecastConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ForecastConfig {
    periods_to_forecast: u32,
    default_model: ModelType,
    fallback_model: ModelType,
    account_overrides: HashMap<String, AccountOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ModelType {
    Auto,
    SimpleAverage,
    LastValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct AccountOverride {
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
fn test_full_structure_from_error() {
    // This is the exact structure from the error log (in array format before normalization)
    let mut response = json!({
        "pnlConfig": {
            "forecastConfig": {
                "periodsToForecast": 12,
                "defaultModel": {"type": "auto"},
                "fallbackModel": {"type": "simpleAverage"},
                "accountOverrides": [
                    {
                        "__key__": "Access Advisors - Sales",
                        "__value__": {
                            "processor": {
                                "model": "mstl",
                                "seasonalPeriods": [12],
                                "trendModel": "ets"
                            }
                        }
                    },
                    {
                        "__key__": "Wages",
                        "__value__": {
                            "processor": {
                                "calculation": [
                                    "sumOfAccounts",
                                    "Access Advisors - Sales",
                                    "multiply",
                                    0.35
                                ]
                            }
                        }
                    },
                    {
                        "__key__": "Tax Provision",
                        "__value__": {
                            "processor": {
                                "taxCalculation": [
                                    "summaryValue",
                                    "Profit Before Tax",
                                    "multiply",
                                    0.28
                                ]
                            }
                        }
                    }
                ]
            }
        }
    });

    println!("=== ORIGINAL (with __key__/__value__ arrays) ===");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    let schema = Config::gemini_schema();

    // Step 1: Normalize (convert arrays to objects)
    normalize_json_response(&mut response);
    println!("\n=== AFTER NORMALIZE ===");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Step 2: Prune null fields
    prune_null_fields(&mut response);
    println!("\n=== AFTER PRUNE_NULL ===");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Step 3: Unflatten enums
    unflatten_externally_tagged_enums(&mut response, &schema);
    println!("\n=== AFTER UNFLATTEN ===");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Step 4: Try to deserialize
    let result: Result<Config, _> = serde_json::from_value(response.clone());
    match &result {
        Ok(config) => {
            println!("\n✅ SUCCESS!");
            println!("Config: {:?}", config);
        }
        Err(e) => {
            println!("\n❌ DESERIALIZATION FAILED: {}", e);
            println!("Final JSON that failed to deserialize:");
            println!("{}", serde_json::to_string_pretty(&response).unwrap());
        }
    }

    assert!(result.is_ok(), "Should deserialize successfully");
}
