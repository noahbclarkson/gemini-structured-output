use gemini_structured_output::schema::{
    normalize_json_response, recover_internally_tagged_enums, GeminiStructured,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ForecastModel {
    Auto,
    Mstl {
        #[serde(default)]
        seasonal_periods: Vec<i32>,
    },
    SimpleAverage,
    LastValue,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Calculation {
    metric: String,
    rate: f64,
    operation: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ModelProcessor {
    model: ForecastModel,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct CalculationProcessor {
    calculation: Calculation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct TaxCalculationProcessor {
    tax_calculation: Calculation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
enum PnlProcessor {
    Model(ModelProcessor),
    Calculation(CalculationProcessor),
    TaxCalculation(TaxCalculationProcessor),
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct AccountOverride {
    processor: PnlProcessor,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ForecastConfig {
    periods_to_forecast: i32,
    default_model: ForecastModel,
    fallback_model: ForecastModel,
    account_overrides: HashMap<String, AccountOverride>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct PnlConfig {
    forecast_config: ForecastConfig,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct RootConfig {
    pnl_config: PnlConfig,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type")]
enum CaseSensitiveModel {
    Mstl,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct CaseSensitiveConfig {
    model: CaseSensitiveModel,
}

#[test]
fn test_forecaster_payload_recovery() {
    let schema = RootConfig::gemini_schema();

    let mut raw_gemini_response = json!({
        "pnlConfig": {
            "forecastConfig": {
                "periodsToForecast": 12,
                "defaultModel": { "type": "auto" },
                "fallbackModel": { "type": "lastValue" },
                "accountOverrides": [
                    {
                        "__key__": "Access Advisors - Sales",
                        "__value__": {
                            "processor": {
                                "model": "mstl"
                            }
                        }
                    },
                    {
                        "__key__": "Tax Provision",
                        "__value__": {
                            "processor": {
                                "taxCalculation": [
                                    "Profit Before Tax",
                                    0.28,
                                    "multiply"
                                ]
                            }
                        }
                    },
                    {
                        "__key__": "Depreciation",
                        "__value__": {
                            "processor": {
                                "model": "simpleAverage"
                            }
                        }
                    }
                ]
            }
        }
    });

    normalize_json_response(&mut raw_gemini_response);
    recover_internally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: RootConfig =
        serde_json::from_value(raw_gemini_response).expect("Deserialization failed");

    let sales = parsed
        .pnl_config
        .forecast_config
        .account_overrides
        .get("Access Advisors - Sales")
        .expect("Sales override missing");
    match &sales.processor {
        PnlProcessor::Model(ModelProcessor {
            model: ForecastModel::Mstl { .. },
        }) => {}
        other => panic!("unexpected processor for Sales: {other:?}"),
    }

    let tax = parsed
        .pnl_config
        .forecast_config
        .account_overrides
        .get("Tax Provision")
        .expect("Tax Provision override missing");
    match &tax.processor {
        PnlProcessor::TaxCalculation(TaxCalculationProcessor { tax_calculation }) => {
            assert_eq!(tax_calculation.metric, "Profit Before Tax");
            assert!((tax_calculation.rate - 0.28).abs() < 1e-6);
            assert_eq!(tax_calculation.operation, "multiply");
        }
        other => panic!("unexpected processor for Tax Provision: {other:?}"),
    }

    let depreciation = parsed
        .pnl_config
        .forecast_config
        .account_overrides
        .get("Depreciation")
        .expect("Depreciation override missing");
    match &depreciation.processor {
        PnlProcessor::Model(ModelProcessor {
            model: ForecastModel::SimpleAverage,
        }) => {}
        other => panic!("unexpected processor for Depreciation: {other:?}"),
    }
}

#[test]
fn test_case_insensitive_tag_recovery() {
    let schema = CaseSensitiveConfig::gemini_schema();

    let mut raw_gemini_response = json!({
        "model": "mstl"
    });

    recover_internally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: CaseSensitiveConfig =
        serde_json::from_value(raw_gemini_response).expect("Deserialization failed");

    assert_eq!(parsed.model, CaseSensitiveModel::Mstl);
}
