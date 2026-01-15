use gemini_rust::Model;
use gemini_structured_output::StructuredClientBuilder;
use gemini_structured_output::schema::GeminiStructured;
use std::env;
use xero_forecasting::config::{FullForecastConfig, PnlProcessor, ForecastModel};

#[tokio::test]
#[ignore]
async fn test_real_xero_forecast_config_with_gemini() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini25Flash)
        .with_default_retries(0)
        .with_default_parse_attempts(1)
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
                    Some(PnlProcessor::Model(ForecastModel::Auto { .. }))
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

#[tokio::test]
async fn test_prune_nulls_from_gemini_flattened_enum() {
    let mut gemini_response_with_nulls = serde_json::json!({
        "pnlConfig": {
            "forecastConfig": {
                "periodsToForecast": 12,
                "defaultModel": "Auto",
                "fallbackModel": "LastValue",
                "accountOverrides": [
                    {
                        "__key__": "Revenue - Sales",
                        "__value__": {
                            "processor": {
                                "model": "Auto",
                                "calculation": null,
                                "taxCalculation": null,
                                "pnlScheduledTransactions": null,
                                "targetSeek": null,
                                "conditionalCalculation": null,
                                "externallyCalculated": null
                            },
                            "isNonCash": false,
                            "constraints": { "nonNegative": false }
                        }
                    },
                    {
                        "__key__": "Cost Account",
                        "__value__": {
                            "processor": {
                                "model": null,
                                "calculation": {
                                    "steps": [
                                        {
                                            "operand": { "constant": 100.0 },
                                            "operator": "multiply"
                                        }
                                    ]
                                },
                                "taxCalculation": null,
                                "pnlScheduledTransactions": null,
                                "targetSeek": null,
                                "conditionalCalculation": null,
                                "externallyCalculated": null
                            },
                            "isNonCash": false,
                            "constraints": { "nonNegative": true }
                        }
                    }
                ]
            }
        },
        "historicalPeriodsToFetch": 24,
        "bsConfig": {
            "accountOverrides": {}
        },
        "fiscalYearEndMonth": 3,
        "cashFlowConfig": {}
    });

    println!("\n📋 Gemini Response (with nulls from flattened enum):");
    println!("{}", serde_json::to_string_pretty(&gemini_response_with_nulls).unwrap());

    gemini_structured_output::schema::normalize_json_response(&mut gemini_response_with_nulls);

    println!("\n🔄 After normalize_json_response (converted arrays to objects):");
    println!("{}", serde_json::to_string_pretty(&gemini_response_with_nulls).unwrap());

    gemini_structured_output::schema::prune_null_fields(&mut gemini_response_with_nulls);

    println!("\n✂️  After prune_null_fields (removed nulls):");
    println!("{}", serde_json::to_string_pretty(&gemini_response_with_nulls).unwrap());

    let schema = FullForecastConfig::gemini_schema();
    gemini_structured_output::schema::recover_internally_tagged_enums(&mut gemini_response_with_nulls, &schema);

    println!("\n🔧 After recover_internally_tagged_enums:");
    println!("{}", serde_json::to_string_pretty(&gemini_response_with_nulls).unwrap());

    let result = serde_json::from_value::<FullForecastConfig>(gemini_response_with_nulls);

    match result {
        Ok(config) => {
            println!("\n✅ Deserialization Successful!");
            println!("\nParsed Config:");
            println!("{}", serde_json::to_string_pretty(&config).unwrap());

            let overrides = &config.pnl_config.forecast_config.account_overrides;
            assert_eq!(overrides.len(), 2, "Expected 2 account overrides");

            let revenue_override = overrides.get("Revenue - Sales").expect("Revenue account should exist");
            assert!(matches!(
                revenue_override.processor,
                Some(PnlProcessor::Model(_))
            ), "Revenue account should have Model processor");

            let cost_override = overrides.get("Cost Account").expect("Cost account should exist");
            assert!(matches!(
                cost_override.processor,
                Some(PnlProcessor::Calculation { .. })
            ), "Cost account should have Calculation processor");

            println!("\n✅ All assertions passed! Null pruning successfully fixed the Gemini response.");
        }
        Err(e) => {
            eprintln!("\n❌ Deserialization Failed: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_prune_nulls_from_flattened_enum() {
    let mut bad_response = serde_json::json!({
        "pnlConfig": {
            "forecastConfig": {
                "periodsToForecast": 12,
                "defaultModel": "Auto",
                "fallbackModel": "LastValue",
                "accountOverrides": [
                    {
                        "__key__": "Revenue - Sales",
                        "__value__": {
                            "processor": {
                                "model": "Auto"
                            },
                            "isNonCash": false,
                            "constraints": { "nonNegative": false }
                        }
                    },
                    {
                        "__key__": "Cost Account",
                        "__value__": {
                            "processor": {
                                "calculation": {
                                    "steps": [
                                        {
                                            "operand": { "constant": 100.0 },
                                            "operator": "multiply"
                                        }
                                    ]
                                }
                            },
                            "isNonCash": false,
                            "constraints": { "nonNegative": true }
                        }
                    }
                ]
            }
        },
        "historicalPeriodsToFetch": 24,
        "bsConfig": {
            "accountOverrides": {}
        },
        "fiscalYearEndMonth": 3,
        "cashFlowConfig": {}
    });

    println!("\n📋 Original Response (with nulls):");
    println!("{}", serde_json::to_string_pretty(&bad_response).unwrap());

    gemini_structured_output::schema::normalize_json_response(&mut bad_response);

    println!("\n🔄 After normalize_json_response:");
    println!("{}", serde_json::to_string_pretty(&bad_response).unwrap());

    gemini_structured_output::schema::prune_null_fields(&mut bad_response);

    println!("\n✂️  After prune_null_fields:");
    println!("{}", serde_json::to_string_pretty(&bad_response).unwrap());

    let schema = FullForecastConfig::gemini_schema();
    gemini_structured_output::schema::recover_internally_tagged_enums(&mut bad_response, &schema);

    println!("\n🔧 After recover_internally_tagged_enums:");
    println!("{}", serde_json::to_string_pretty(&bad_response).unwrap());

    let result = serde_json::from_value::<FullForecastConfig>(bad_response);

    match result {
        Ok(config) => {
            println!("\n✅ Deserialization Successful!");
            println!("\nParsed Config:");
            println!("{}", serde_json::to_string_pretty(&config).unwrap());

            let overrides = &config.pnl_config.forecast_config.account_overrides;
            assert_eq!(overrides.len(), 2, "Expected 2 account overrides");

            let revenue_override = overrides.get("Revenue - Sales").expect("Revenue account should exist");
            assert!(matches!(
                revenue_override.processor,
                Some(PnlProcessor::Model(_))
            ), "Revenue account should have Model processor");

            let cost_override = overrides.get("Cost Account").expect("Cost account should exist");
            assert!(matches!(
                cost_override.processor,
                Some(PnlProcessor::Calculation { .. })
            ), "Cost account should have Calculation processor");

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Deserialization Failed: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_unflatten_nested_enums_real_data() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("trace")
        .with_test_writer()
        .try_init();

    let mut gemini_response = serde_json::json!({
        "pnlConfig": {
            "forecastConfig": {
                "periodsToForecast": 12,
                "defaultModel": "Auto",
                "fallbackModel": "LastValue",
                "accountOverrides": {
                    "Revenue - Sales": {
                        "processor": {
                            "model": "mstl",
                            "seasonalPeriods": [12],
                            "trendModel": "ets"
                        },
                        "isNonCash": false,
                        "constraints": { "nonNegative": true }
                    },
                    "Cost Account": {
                        "processor": {
                            "calculation": {
                                "steps": [
                                    {
                                        "operand": { "constant": 100.0 },
                                        "operator": "multiply"
                                    }
                                ]
                            }
                        },
                        "isNonCash": false,
                        "constraints": { "nonNegative": false }
                    }
                }
            }
        },
        "historicalPeriodsToFetch": 24,
        "bsConfig": {
            "accountOverrides": {}
        },
        "fiscalYearEndMonth": 3,
        "cashFlowConfig": {}
    });

    println!("\n📋 Original Gemini Response (flattened nested enum):");
    println!("{}", serde_json::to_string_pretty(&gemini_response).unwrap());

    let schema = FullForecastConfig::gemini_schema();

    println!("\n🔍 Top-level schema keys: {:?}", schema.as_object().map(|o| o.keys().collect::<Vec<_>>()));

    gemini_structured_output::schema::normalize_json_response(&mut gemini_response);
    println!("\n🔄 After normalize:");
    println!("{}", serde_json::to_string_pretty(&gemini_response).unwrap());

    gemini_structured_output::schema::prune_null_fields(&mut gemini_response);
    println!("\n✂️  After prune_nulls:");
    println!("{}", serde_json::to_string_pretty(&gemini_response).unwrap());

    gemini_structured_output::schema::unflatten_externally_tagged_enums(&mut gemini_response, &schema);
    println!("\n🔧 After unflatten_externally_tagged_enums:");
    println!("{}", serde_json::to_string_pretty(&gemini_response).unwrap());

    gemini_structured_output::schema::recover_internally_tagged_enums(&mut gemini_response, &schema);
    println!("\n🎯 After recover_internally_tagged_enums:");
    println!("{}", serde_json::to_string_pretty(&gemini_response).unwrap());

    let result = serde_json::from_value::<FullForecastConfig>(gemini_response);

    match result {
        Ok(config) => {
            println!("\n✅ Deserialization Successful!");

            let overrides = &config.pnl_config.forecast_config.account_overrides;
            assert_eq!(overrides.len(), 2, "Expected 2 account overrides");

            let revenue_override = overrides.get("Revenue - Sales").expect("Revenue account should exist");
            assert!(matches!(
                revenue_override.processor,
                Some(PnlProcessor::Model(ForecastModel::Mstl { .. }))
            ), "Revenue account should have Mstl model, got: {:?}", revenue_override.processor);

            let cost_override = overrides.get("Cost Account").expect("Cost account should exist");
            assert!(matches!(
                cost_override.processor,
                Some(PnlProcessor::Calculation { .. })
            ), "Cost account should have Calculation processor");

            println!("\n✅ All assertions passed! Nested enum unflattening works!");
        }
        Err(e) => {
            eprintln!("\n❌ Deserialization Failed: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

#[test]
fn test_prune_null_fields_basic() {
    let mut value = serde_json::json!({
        "name": "John",
        "age": null,
        "nested": {
            "field1": "value",
            "field2": null,
            "field3": {
                "deep": null
            }
        },
        "array": [
            {"a": 1, "b": null},
            {"a": null, "b": 2}
        ]
    });

    gemini_structured_output::schema::prune_null_fields(&mut value);

    let expected = serde_json::json!({
        "name": "John",
        "nested": {
            "field1": "value",
            "field3": {}
        },
        "array": [
            {"a": 1},
            {"b": 2}
        ]
    });

    assert_eq!(value, expected, "Null fields should be removed recursively");
}
