//! Example: Dynamic/runtime schema generation.
//!
//! Build a JSON Schema at runtime (e.g., from user-provided headers) and
//! generate structured output without defining a Rust struct in advance.

use gemini_structured_output::prelude::*;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let headers = vec!["name", "email", "department"];

    // Build a simple object schema dynamically.
    let mut properties = serde_json::Map::new();
    for header in &headers {
        properties.insert(
            header.to_string(),
            json!({
                "type": "string",
                "description": format!("Field extracted for header '{header}'"),
            }),
        );
    }

    let dynamic_schema = json!({
        "type": "object",
        "properties": properties,
        "required": headers,
        "additionalProperties": false
    });

    // Mock handler keeps this example offline-friendly.
    let client = StructuredClientBuilder::new("mock-key")
        .with_mock(|req: MockRequest| {
            if req.target == "serde_json::value::Value" {
                return Ok(r#"{
                        "name": "Ada Lovelace",
                        "email": "ada@example.com",
                        "department": "Research"
                    }"#
                .to_string());
            }
            Err(StructuredError::Context(format!(
                "Unexpected target for dynamic schema: {}",
                req.target
            )))
        })
        .build()?;

    let ctx = ContextBuilder::new()
        .with_system("Fill out the JSON object using the provided schema.")
        .add_user_text("Employee record to extract:");

    let value: Value = client.generate_dynamic(dynamic_schema, ctx, None).await?;

    println!("Dynamic schema result (Value): {value:#}");
    Ok(())
}
