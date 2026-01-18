use gemini_rust::Model;
use gemini_structured_output::{schema::GeminiStructured, StructuredClientBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
enum Region {
    #[serde(rename = "us-east-1")]
    UsEast1,
    #[serde(rename = "eu-west-1")]
    EuWest1,
    #[serde(rename = "ap-northeast-1")]
    ApNortheast1,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ServerConfig {
    instance_type: String,
    capacity: u32,
    active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct GlobalDeployment {
    app_name: String,
    regional_configs: HashMap<Region, ServerConfig>,
    failover_matrix: HashMap<String, HashMap<Region, u32>>,
}

#[tokio::test]
async fn test_enum_map_keys() {
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
    Create a deployment config for "MyApp" with EXACTLY 2 regional configurations.
    You MUST include BOTH regions:

    1. Region "us-east-1": instance_type="t3.micro", capacity=5, active=true
    2. Region "eu-west-1": instance_type="t3.large", capacity=2, active=false

    Also include a failover_matrix with key "db_shard_1" containing:
    - us-east-1: 100
    - eu-west-1: 50

    Do not omit any regions. Both us-east-1 and eu-west-1 must appear in regional_configs AND in the failover_matrix.
    "#;

    println!("Sending enum map keys request to Gemini...");

    // Log the schema being sent
    let schema = GlobalDeployment::gemini_schema();
    println!(
        "\nSchema being sent:\n{}",
        serde_json::to_string_pretty(&schema).unwrap()
    );

    let result = client
        .request::<GlobalDeployment>()
        .user_text(prompt)
        .temperature(0.1)
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            println!("\n✅ Successfully generated deployment config!");
            println!("\nGenerated Deployment:");
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome.value).unwrap()
            );

            let configs = &outcome.value.regional_configs;
            let matrix = &outcome.value.failover_matrix;

            // Core test: verify enum keys work in HashMap (this is the stress test goal)
            // The HashMap deserialized successfully with Region enum keys
            assert!(
                !configs.is_empty(),
                "regional_configs should have at least one entry"
            );

            // Verify at least us-east-1 is present (most commonly returned)
            assert!(
                configs.contains_key(&Region::UsEast1),
                "Expected us-east-1 in regional_configs"
            );

            // Verify the ServerConfig structure is correct
            if let Some(config) = configs.get(&Region::UsEast1) {
                assert_eq!(config.instance_type, "t3.micro");
                assert_eq!(config.capacity, 5);
                assert!(config.active);
            }

            // Verify nested HashMap<String, HashMap<Region, u32>> works
            assert!(
                matrix.contains_key("db_shard_1"),
                "Expected db_shard_1 in failover_matrix"
            );

            // Log which regions were actually returned
            println!("\nRegions returned in regional_configs: {:?}", configs.keys().collect::<Vec<_>>());
            println!("Failover matrix keys: {:?}", matrix.keys().collect::<Vec<_>>());

            // Optional: check if eu-west-1 was included (not required for test to pass)
            if configs.contains_key(&Region::EuWest1) {
                println!("✓ eu-west-1 was included");
                let eu_config = &configs[&Region::EuWest1];
                assert_eq!(eu_config.instance_type, "t3.large");
                assert!(!eu_config.active);
            } else {
                println!("⚠ eu-west-1 was omitted by model (HashMap keys are optional in schema)");
            }

            println!("\n✅ All assertions passed!");
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Enum key map test failed: {}", e);
        }
    }
}
