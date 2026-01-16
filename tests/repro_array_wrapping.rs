use gemini_structured_output::schema::{unflatten_externally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "PascalCase")]
enum Processor {
    Calculation { steps: Vec<Value> },
    Model { period: u32, trend: String },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Config {
    processor: Processor,
}

#[test]
fn test_recover_array_into_single_vec_field() {
    let schema = Config::gemini_schema();
    let mut raw_gemini_response = json!({
        "processor": ["add", 100, "multiply"]
    });

    unflatten_externally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: Config = serde_json::from_value(raw_gemini_response)
        .expect("expected array to map into single Vec field");

    match parsed.processor {
        Processor::Calculation { steps } => {
            assert_eq!(steps.len(), 3);
            assert_eq!(steps[0], json!("add"));
            assert_eq!(steps[1], json!(100));
        }
        _ => panic!("expected Calculation variant"),
    }
}

#[test]
fn test_array_matches_multi_field_variant() {
    let schema = Config::gemini_schema();
    let mut raw_gemini_response = json!({
        "processor": [12, "ets"]
    });

    unflatten_externally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: Config = serde_json::from_value(raw_gemini_response)
        .expect("expected array to map into multi-field variant");

    match parsed.processor {
        Processor::Model { period, trend } => {
            assert_eq!(period, 12);
            assert_eq!(trend, "ets");
        }
        _ => panic!("expected Model variant"),
    }
}
