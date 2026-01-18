use gemini_rust::{Gemini, GenerationConfig, Model};
use gemini_structured_output::schema::{clean_schema_for_gemini, strip_x_fields, GeminiStructured};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
#[ignore = "Probes strict vs prompt schema behavior, requires key"]
async fn probe_enum_map_keys_strict_vs_prompt() {
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model = Model::Gemini3Flash;
    let client = Gemini::with_model(&api_key, model).expect("Failed to create client");

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

    let mut schema = GlobalDeployment::gemini_schema();
    clean_schema_for_gemini(&mut schema);
    strip_x_fields(&mut schema);

    let schema_text = serde_json::to_string_pretty(&schema).unwrap_or_default();
    let schema_instruction = format!(
        "You must output valid JSON matching this schema exactly:\n{}",
        schema_text
    );

    print_required_keys(&schema);

    let modes = [
        ("strict_schema", true),
        ("prompt_schema", false),
    ];

    for (name, strict) in modes {
        println!("\n=== Mode: {} ===", name);

        let mut config = GenerationConfig::default();
        config.response_mime_type = Some("application/json".to_string());
        if strict {
            config.response_json_schema = Some(schema.clone());
        }

        let mut builder = client.generate_content().with_user_message(prompt);
        if !strict {
            builder = builder.with_system_instruction(schema_instruction.clone());
        }

        let result = builder.with_generation_config(config).execute().await;
        match result {
            Ok(resp) => {
                let text = resp.text();
                println!("Raw response:\n{}", text);

                match parse_json(&text) {
                    Ok(value) => {
                        match validate_schema(&schema, &value) {
                            Ok(()) => println!("Schema validation: ok"),
                            Err(err) => println!("Schema validation errors: {}", err),
                        }
                        report_missing_keys(&value);
                    }
                    Err(err) => {
                        println!("Failed to parse JSON: {}", err);
                    }
                }
            }
            Err(e) => {
                println!("Request failed: {:?}", e);
            }
        }
    }
}

fn report_missing_keys(value: &Value) {
    let required_regions = ["us-east-1", "eu-west-1"];

    let configs = value
        .get("regional_configs")
        .and_then(|v| v.as_object());
    let missing_configs: Vec<&str> = required_regions
        .iter()
        .copied()
        .filter(|region| {
            configs
                .map(|obj| !obj.contains_key(*region))
                .unwrap_or(true)
        })
        .collect();

    let matrix = value
        .get("failover_matrix")
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("db_shard_1"))
        .and_then(|v| v.as_object());
    let missing_matrix: Vec<&str> = required_regions
        .iter()
        .copied()
        .filter(|region| {
            matrix
                .map(|obj| !obj.contains_key(*region))
                .unwrap_or(true)
        })
        .collect();

    println!(
        "Missing in regional_configs: {:?}",
        if missing_configs.is_empty() {
            vec!["none"]
        } else {
            missing_configs
        }
    );
    println!(
        "Missing in failover_matrix.db_shard_1: {:?}",
        if missing_matrix.is_empty() {
            vec!["none"]
        } else {
            missing_matrix
        }
    );
}

fn validate_schema(schema: &Value, value: &Value) -> Result<(), String> {
    let validator = jsonschema::validator_for(schema).map_err(|e| e.to_string())?;
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|err| format!("{}: {}", err.instance_path(), err))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn print_required_keys(schema: &Value) {
    let regional_required = schema
        .pointer("/properties/regional_configs/required")
        .cloned()
        .unwrap_or(Value::Null);
    let matrix_required = schema
        .pointer("/properties/failover_matrix/additionalProperties/required")
        .cloned()
        .unwrap_or(Value::Null);

    println!("Schema required for regional_configs: {}", regional_required);
    println!(
        "Schema required for failover_matrix values: {}",
        matrix_required
    );
}

fn parse_json(text: &str) -> Result<Value, serde_json::Error> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Ok(value);
    }

    if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed.rfind("```") {
            if start < end {
                if let Some(newline) = trimmed[start..end].find('\n') {
                    let content_start = start + newline + 1;
                    if content_start < end {
                        let inner = trimmed[content_start..end].trim();
                        if let Ok(value) = serde_json::from_str::<Value>(inner) {
                            return Ok(value);
                        }
                    }
                }
            }
        }
    }

    if let Some(start) = trimmed.find(['{', '[']) {
        if let Some(end) = trimmed.rfind(['}', ']']) {
            if start <= end {
                let slice = &trimmed[start..=end];
                return serde_json::from_str::<Value>(slice);
            }
        }
    }

    serde_json::from_str::<Value>(trimmed)
}
