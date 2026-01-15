use gemini_structured_output::schema::{strip_x_fields, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct TestConfig {
    settings: HashMap<String, TestSetting>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
enum TestSetting {
    Option1 { value: String },
    Option2 { count: i32 },
}

#[test]
fn test_x_fields_preserved_in_schema() {
    let schema = TestConfig::gemini_schema();

    // Find the settings field which should be a HashMap
    let settings_schema = schema
        .get("properties")
        .and_then(|p| p.get("settings"))
        .expect("Should have settings property");

    // The schema should have been transformed to type: array with x-additionalProperties-original
    assert_eq!(
        settings_schema.get("type").and_then(|t| t.as_str()),
        Some("array"),
        "HashMap should be transformed to array type"
    );

    // The x-additionalProperties-original field should be present
    let x_field = settings_schema.get("x-additionalProperties-original");
    assert!(
        x_field.is_some(),
        "Schema should preserve x-additionalProperties-original for unflattening"
    );

    // The x-field should contain anyOf for the enum variants
    let any_of = x_field
        .and_then(|x| x.get("anyOf").or_else(|| x.get("x-anyOf-original")));
    assert!(
        any_of.is_some(),
        "x-additionalProperties-original should contain anyOf or x-anyOf-original for enum variants"
    );

    println!("✅ Schema correctly preserves x-additionalProperties-original");
}

#[test]
fn test_strip_x_fields_removes_all_x_fields() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "field1": {
                "type": "string",
                "x-custom": "should be removed"
            },
            "field2": {
                "type": "array",
                "x-additionalProperties-original": {
                    "type": "object"
                },
                "items": {
                    "type": "object",
                    "x-anyOf-original": ["variant1", "variant2"]
                }
            }
        },
        "x-root-field": "should also be removed"
    });

    strip_x_fields(&mut schema);

    // Verify all x-* fields are removed
    assert!(
        !has_x_fields(&schema),
        "All x-* fields should be removed after strip_x_fields"
    );

    // Verify other fields are preserved
    assert!(
        schema.get("type").is_some(),
        "Non-x fields should be preserved"
    );
    assert!(
        schema.get("properties").is_some(),
        "Properties should be preserved"
    );

    println!("✅ strip_x_fields correctly removes all x-* custom fields");
}

fn has_x_fields(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            for (key, val) in map {
                if key.starts_with("x-") {
                    return true;
                }
                if has_x_fields(val) {
                    return true;
                }
            }
            false
        }
        Value::Array(arr) => arr.iter().any(has_x_fields),
        _ => false,
    }
}
