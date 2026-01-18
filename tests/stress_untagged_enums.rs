use gemini_rust::Model;
use gemini_structured_output::StructuredClientBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
enum ConfigurationValue {
    // Order matters for untagged deserialization!
    Boolean(bool),
    Integer(i64),
    String(String),
    Array(Vec<String>),
    Object(InnerConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct InnerConfig {
    enabled: bool,
    host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MixedConfig {
    settings: Vec<ConfigurationValue>,
}

#[tokio::test]
async fn test_untagged_enum_polymorphism() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .with_test_writer()
        .init();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = StructuredClientBuilder::new(api_key)
        .with_model(Model::Gemini3Flash)
        .with_default_retries(0)
        .build()
        .expect("Failed to build client");

    let prompt = r#"
    Generate a settings list with exactly these 5 items in this order:
    1. A boolean true.
    2. The integer 42.
    3. The string "production".
    4. An array ["us-west", "us-east"].
    5. An object { "enabled": false, "host": "localhost" }.
    "#;

    println!("Sending untagged enum polymorphism request to Gemini...");

    let result = client
        .request::<MixedConfig>()
        .user_text(prompt)
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            println!("\n✅ Successfully generated mixed config!");
            println!("\nGenerated Mixed Config:");
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome.value).unwrap()
            );

            let s = &outcome.value.settings;
            assert_eq!(s.len(), 5);

            assert!(matches!(s[0], ConfigurationValue::Boolean(true)));
            assert!(matches!(s[1], ConfigurationValue::Integer(42)));
            assert!(matches!(s[2], ConfigurationValue::String(ref v) if v == "production"));
            assert!(matches!(s[3], ConfigurationValue::Array(_)));
            assert!(matches!(s[4], ConfigurationValue::Object(_)));

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Untagged enum test failed: {}", e);
        }
    }
}
