use gemini_rust::{Gemini, GenerationConfig, GenerationResponse, Model, Part};
use serde_json::{json, Value};
use std::env;

#[tokio::test]
#[ignore = "Probes API capabilities, requires key"]
async fn test_probe_map_capabilities() {
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model = Model::Gemini3Flash;
    let client = Gemini::with_model(&api_key, model).expect("Failed to create client");

    let simple_map_schema = json!({
        "type": "object",
        "properties": {
            "dictionary": {
                "type": "object",
                "additionalProperties": { "type": "string" }
            }
        },
        "required": ["dictionary"],
        "additionalProperties": false
    });

    let complex_map_schema = json!({
        "type": "object",
        "properties": {
            "configs": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "host": { "type": "string" },
                        "port": { "type": "integer" }
                    },
                    "required": ["host", "port"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["configs"],
        "additionalProperties": false
    });

    let schemas = vec![
        ("Simple Map", simple_map_schema),
        ("Complex Map", complex_map_schema),
    ];

    for (name, schema) in schemas {
        let modes = [("strict_schema", true), ("prompt_schema", false)];

        for (mode, strict) in modes {
            println!("Testing schema: {} ({})", name, mode);
            let mut config = GenerationConfig {
                response_mime_type: Some("application/json".to_string()),
                ..Default::default()
            };
            if strict {
                config.response_json_schema = Some(schema.clone());
            }

            let mut builder = client
                .generate_content()
                .with_generation_config(config)
                .with_user_message("Generate example data for this schema with at least 2 entries.");

            if !strict {
                let schema_instruction = format!(
                    "You must output valid JSON matching this schema exactly:\n{}",
                    serde_json::to_string_pretty(&schema).unwrap_or_default()
                );
                builder = builder.with_system_instruction(schema_instruction);
            }

            let result = builder.execute().await;

            match result {
                Ok(resp) => {
                    let text = resp.text();
                    if text.trim().is_empty() {
                        println!("EMPTY {} ({}): Response text was empty", name, mode);
                        log_response_details(&resp);
                        println!("------------------------------------------------");
                        continue;
                    }

                    println!("OK {} ({}): Success\n{}", name, mode, text);
                    match parse_json(&text) {
                        Ok(value) => match validate_schema(&schema, &value) {
                            Ok(()) => println!("Validation: ok"),
                            Err(err) => println!("Validation errors: {}", err),
                        },
                        Err(err) => {
                            println!("Parse error: {}", err);
                            log_response_details(&resp);
                        }
                    }
                }
                Err(e) => println!("ERR {} ({}): Failed\n{:?}", name, mode, e),
            }
            println!("------------------------------------------------");
        }
    }
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

fn log_response_details(resp: &GenerationResponse) {
    println!("Response details: {:#?}", resp.prompt_feedback);
    for (idx, candidate) in resp.candidates.iter().enumerate() {
        println!("Candidate {}: finish_reason={:?}", idx, candidate.finish_reason);
        if let Some(role) = &candidate.content.role {
            println!("Candidate {}: role={:?}", idx, role);
        }
        match candidate.content.parts.as_ref() {
            Some(parts) => {
                for (pidx, part) in parts.iter().enumerate() {
                    let kind = match part {
                        Part::Text { .. } => "Text",
                        Part::InlineData { .. } => "InlineData",
                        Part::FileData { .. } => "FileData",
                        Part::FunctionCall { .. } => "FunctionCall",
                        Part::FunctionResponse { .. } => "FunctionResponse",
                        Part::ExecutableCode { .. } => "ExecutableCode",
                        Part::CodeExecutionResult { .. } => "CodeExecutionResult",
                    };
                    println!("Candidate {} part {}: {}", idx, pidx, kind);
                }
            }
            None => println!("Candidate {}: no parts", idx),
        }
    }
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
