use serde_json::json;

#[test]
fn test_x_additionalproperties_preservation() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("trace")
        .with_test_writer()
        .try_init();

    // Create a simple schema with additionalProperties
    let mut schema = json!({
        "type": "object",
        "additionalProperties": {
            "type": "string"
        }
    });

    println!("\n=== Before clean_schema_for_gemini ===");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    // Check the structure
    println!("\n=== Schema structure check ===");
    println!("Has 'type': {}", schema.get("type").is_some());
    println!("Type value: {:?}", schema.get("type"));
    println!("Has 'additionalProperties': {}", schema.get("additionalProperties").is_some());

    // Apply Gemini transformations
    gemini_structured_output::schema::clean_schema_for_gemini(&mut schema);

    println!("\n=== After clean_schema_for_gemini ===");
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    // Check if x-additionalProperties-original was added
    println!("\n=== Checking for x-additionalProperties-original ===");
    println!("Has x-additionalProperties-original: {}", schema.get("x-additionalProperties-original").is_some());

    if let Some(x_add_props) = schema.get("x-additionalProperties-original") {
        println!("Value: {}", serde_json::to_string_pretty(x_add_props).unwrap());
        assert!(true, "x-additionalProperties-original was successfully added!");
    } else {
        panic!("x-additionalProperties-original was NOT added!");
    }
}
