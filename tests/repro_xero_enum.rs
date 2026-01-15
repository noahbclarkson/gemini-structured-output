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
    Ets,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Config {
    // This creates the HashMap -> Array<KV> transformation
    overrides: HashMap<String, AccountConfig>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct AccountConfig {
    processor: Processor,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Processor {
    model: ForecastModel,
}

#[test]
fn test_xero_enum_recovery() {
    let schema = Config::gemini_schema();

    // Gemini collapses internally tagged enums to string literals.
    let mut raw_gemini_response = json!({
        "overrides": [
            {
                "__key__": "Sales",
                "__value__": {
                    "processor": {
                        "model": "mstl"
                    }
                }
            },
            {
                "__key__": "Rent",
                "__value__": {
                    "processor": {
                        "model": "auto"
                    }
                }
            }
        ]
    });

    // Normalize the KV-array back into a map.
    normalize_json_response(&mut raw_gemini_response);

    recover_internally_tagged_enums(&mut raw_gemini_response, &schema);

    let parsed: Config =
        serde_json::from_value(raw_gemini_response).expect("Deserialization failed");

    assert!(matches!(
        parsed.overrides.get("Sales").unwrap().processor.model,
        ForecastModel::Mstl { .. }
    ));
    assert!(matches!(
        parsed.overrides.get("Rent").unwrap().processor.model,
        ForecastModel::Auto
    ));
}
