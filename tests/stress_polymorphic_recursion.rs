use gemini_rust::Model;
use gemini_structured_output::StructuredClientBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
enum FileSystemNode {
    File {
        name: String,
        size: u64,
        attributes: HashMap<String, FileAttribute>,
    },
    Directory {
        name: String,
        children: Vec<FileSystemNode>,
        metadata: DirectoryMetadata,
    },
    Symlink {
        source: String,
        target: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", content = "value")]
enum FileAttribute {
    Permissions(String),
    Hidden(bool),
    Tags(Vec<String>),
    Custom { key: String, val: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct DirectoryMetadata {
    owner: String,
    last_modified: String,
    quota: Option<Box<DirectoryMetadata>>,
}

#[tokio::test]
async fn test_polymorphic_recursion() {
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
    Generate a virtual file system structure representing a web server project.

    1. Root directory "var".
    2. Inside "var", a directory "www" owned by "root".
    3. Inside "www", a file "index.html" (size 1024) with attributes:
       - permissions: "644"
       - tags: ["html", "public"]
    4. Inside "www", a symlink "latest" pointing to "v2".
    5. Ensure the directory metadata is populated.
    "#;

    println!("Sending polymorphic recursion request to Gemini...");

    let result = client
        .request::<FileSystemNode>()
        .user_text(prompt)
        .temperature(0.1)
        .execute()
        .await;

    match result {
        Ok(outcome) => {
            println!("\n✅ Successfully generated file system structure!");
            println!("\nGenerated FS:");
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome.value).unwrap()
            );

            if let FileSystemNode::Directory { name, children, .. } = outcome.value {
                assert_eq!(name, "var");
                let www = children.iter().find(|c| {
                    matches!(c, FileSystemNode::Directory { name, .. } if name == "www")
                });
                assert!(www.is_some(), "www directory missing");
                println!("\n✅ All assertions passed!");
            } else {
                panic!("Root was not a directory");
            }
        }
        Err(e) => {
            eprintln!("\n❌ Request failed with error: {:?}", e);
            eprintln!("\nFull error details: {:#?}", e);
            panic!("Recursion test failed: {}", e);
        }
    }
}
