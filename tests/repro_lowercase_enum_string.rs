use gemini_structured_output::schema::{unflatten_externally_tagged_enums, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "PascalCase")]
enum Model {
    Auto,
    Mstl {
        seasonal_periods: Vec<u32>,
        trend_model: String,
    },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Config {
    model: Model,
}

#[test]
fn test_lowercase_enum_string_coerces_to_variant() {
    let schema = Config::gemini_schema();
    let mut raw = json!({ "model": "mstl" });

    unflatten_externally_tagged_enums(&mut raw, &schema);

    let parsed: Config =
        serde_json::from_value(raw).expect("expected lowercase string to deserialize");

    match parsed.model {
        Model::Mstl {
            seasonal_periods,
            trend_model,
        } => {
            assert!(seasonal_periods.is_empty());
            assert_eq!(trend_model, "");
        }
        _ => panic!("expected Mstl variant"),
    }
}
