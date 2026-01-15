use gemini_structured_output::schema::{recover_internally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ForecastModel {
    Auto,
    Mstl {
        #[serde(default)]
        seasonal_periods: Vec<i32>,
    },
    Ets,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
enum Processor {
    Model { model: ForecastModel },
    Raw(String),
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Config {
    processor: Processor,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct TaxCalculation {
    method: String,
    rate: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct TaxConfig {
    tax_calculation: TaxCalculation,
}

#[test]
fn test_anyof_object_recursion_recovers_nested_enum() {
    let schema = Config::gemini_schema();
    let processor_schema = schema
        .get("properties")
        .and_then(|p| p.get("processor"))
        .expect("processor schema should exist");
    assert!(
        processor_schema.get("anyOf").is_some(),
        "processor schema should retain anyOf for mixed variants"
    );

    let mut raw_gemini_response = json!({
        "processor": {
            "model": "mstl"
        }
    });

    recover_internally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: Config =
        serde_json::from_value(raw_gemini_response).expect("Deserialization failed");

    match parsed.processor {
        Processor::Model {
            model: ForecastModel::Mstl { .. },
        } => {}
        other => panic!("unexpected processor variant: {other:?}"),
    }
}

#[test]
fn test_array_to_object_recovery() {
    let schema = TaxConfig::gemini_schema();
    let mut raw_gemini_response = json!({
        "tax_calculation": ["Profit", 0.28]
    });

    recover_internally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: TaxConfig =
        serde_json::from_value(raw_gemini_response).expect("Deserialization failed");

    assert_eq!(parsed.tax_calculation.method, "Profit");
    assert!((parsed.tax_calculation.rate - 0.28).abs() < 1e-6);
}
