use gemini_rust::{Gemini, GenerationConfig, Model};
use serde_json::{json, Map, Value};
use std::env;

#[tokio::test]
#[ignore = "Probes enum-map schema vs map-like schema, requires key"]
async fn probe_enum_map_schema_transform() {
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model = Model::Gemini3Flash;
    let client = Gemini::with_model(&api_key, model).expect("Failed to create client");

    let original_schema = json!({
        "title": "GlobalDeployment",
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string"
            },
            "regional_configs": {
                "type": "object",
                "properties": {
                    "ap-northeast-1": {
                        "$ref": "#/$defs/ServerConfig"
                    },
                    "eu-west-1": {
                        "$ref": "#/$defs/ServerConfig"
                    },
                    "us-east-1": {
                        "$ref": "#/$defs/ServerConfig"
                    }
                },
                "additionalProperties": false
            },
            "failover_matrix": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "ap-northeast-1": {
                            "type": "integer",
                            "format": "uint32",
                            "minimum": 0
                        },
                        "eu-west-1": {
                            "type": "integer",
                            "format": "uint32",
                            "minimum": 0
                        },
                        "us-east-1": {
                            "type": "integer",
                            "format": "uint32",
                            "minimum": 0
                        }
                    },
                    "additionalProperties": false
                }
            }
        },
        "required": [
            "app_name",
            "regional_configs",
            "failover_matrix"
        ],
        "$defs": {
            "ServerConfig": {
                "type": "object",
                "properties": {
                    "instance_type": {
                        "type": "string"
                    },
                    "capacity": {
                        "type": "integer",
                        "format": "uint32",
                        "minimum": 0
                    },
                    "active": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "instance_type",
                    "capacity",
                    "active"
                ]
            }
        }
    });

    let transformed_schema = transform_enum_maps_to_additional_properties(&original_schema);

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

    let cases = [
        ("enum_keys_schema", original_schema),
        ("map_like_schema", transformed_schema),
    ];

    for (name, schema) in cases {
        println!("\n=== Schema: {} ===", name);
        println!("Schema:\n{}", serde_json::to_string_pretty(&schema).unwrap());

        let config = GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_json_schema: Some(schema.clone()),
            ..Default::default()
        };

        let result = client
            .generate_content()
            .with_generation_config(config)
            .with_user_message(prompt)
            .execute()
            .await;

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
                    Err(err) => println!("Failed to parse JSON: {}", err),
                }
            }
            Err(e) => println!("Request failed: {:?}", e),
        }
    }
}

fn transform_enum_maps_to_additional_properties(schema: &Value) -> Value {
    let mut out = schema.clone();

    if let Some(regional) = out
        .pointer_mut("/properties/regional_configs")
        .and_then(Value::as_object_mut)
    {
        collapse_properties_to_additional_properties(regional);
    }

    if let Some(inner) = out
        .pointer_mut("/properties/failover_matrix/additionalProperties")
        .and_then(Value::as_object_mut)
    {
        collapse_properties_to_additional_properties(inner);
    }

    out
}

fn collapse_properties_to_additional_properties(obj: &mut Map<String, Value>) -> bool {
    let props = match obj.get("properties").and_then(|v| v.as_object()) {
        Some(props) => props,
        None => return false,
    };

    let mut values = props.values();
    let first = match values.next() {
        Some(value) => value.clone(),
        None => return false,
    };

    if values.any(|value| value != &first) {
        return false;
    }

    obj.insert("type".to_string(), Value::String("object".to_string()));
    obj.insert("additionalProperties".to_string(), first);
    obj.remove("properties");
    true
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
