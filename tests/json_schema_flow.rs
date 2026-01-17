use gemini_structured_output::schema::{
    clean_schema_for_gemini, coerce_enum_strings, recover_internally_tagged_enums, strip_x_fields,
    GeminiStructured,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
enum Mode {
    Auto,
    Manual { threshold: i32 },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Config {
    mode: Mode,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Activity {
    Operating,
    Investing,
    Financing,
}

#[test]
fn clean_schema_strips_unsupported_keywords() {
    let mut schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Config",
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "examples": ["auto"],
                "readOnly": true
            }
        }
    });

    clean_schema_for_gemini(&mut schema);

    assert!(schema.get("$schema").is_none());
    assert!(schema.get("title").is_none());
    assert!(
        schema
            .get("properties")
            .and_then(|p| p.get("mode"))
            .and_then(|v| v.get("examples"))
            .is_none()
    );
}

#[test]
fn strip_x_fields_removes_internal_markers() {
    let mut schema = json!({
        "type": "object",
        "x-debug": true,
        "properties": {
            "mode": {"type": "string", "x-extra": "value"}
        }
    });

    strip_x_fields(&mut schema);

    assert!(schema.get("x-debug").is_none());
    assert!(
        schema
            .get("properties")
            .and_then(|p| p.get("mode"))
            .and_then(|m| m.get("x-extra"))
            .is_none()
    );
}

#[test]
fn recover_internally_tagged_enum_from_string() {
    let schema = Config::gemini_schema();
    let mut value = json!({"mode": "auto"});

    recover_internally_tagged_enums(&mut value, &schema);

    let parsed: Config = serde_json::from_value(value).expect("deserialization should succeed");
    assert_eq!(parsed.mode, Mode::Auto);
}

#[test]
fn coerce_enum_string_values() {
    let schema = Activity::gemini_schema();
    let mut value = json!("Operating Activities");

    coerce_enum_strings(&mut value, &schema);

    let parsed: Activity = serde_json::from_value(value).expect("deserialization should succeed");
    assert_eq!(parsed, Activity::Operating);
}
