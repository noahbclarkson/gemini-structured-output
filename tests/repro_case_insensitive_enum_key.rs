use gemini_structured_output::schema::{unflatten_externally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
enum InnerConfig {
    Auto(InnerAuto),
    Manual(InnerManual),
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct InnerAuto {
    drivers: Vec<String>,
    lags: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct InnerManual {
    drivers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
enum Outer {
    Example(InnerConfig),
    LastValue,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Config {
    processor: Outer,
}

#[test]
fn test_case_insensitive_variant_key_coerces_newtype_payload() {
    let schema = Config::gemini_schema();
    let mut raw = json!({
        "processor": {
            "Example": "not-a-variant"
        }
    });

    unflatten_externally_tagged_enums(&mut raw, &schema);

    let parsed: Config = serde_json::from_value(raw).expect("should deserialize");
    assert!(matches!(parsed.processor, Outer::Example(_)));
}
