# gemini-structured-output

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/gemini-structured-output.svg)](https://crates.io/crates/gemini-structured-output)
[![Documentation](https://docs.rs/gemini-structured-output/badge.svg)](https://docs.rs/gemini-structured-output)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Gemini API](https://img.shields.io/badge/Gemini-API-8E75B2)](https://ai.google.dev/)

**Production-grade structured generation, self-correcting refinement loops, and type-safe agentic workflows for Google Gemini.**

</div>

---

## 📖 Table of Contents

- [gemini-structured-output](#gemini-structured-output)
  - [📖 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Core Concepts](#core-concepts)
    - [Structured Generation](#structured-generation)
    - [The Refinement Engine](#the-refinement-engine)
    - [Context \& Files](#context--files)
    - [Caching](#caching)
  - [Workflow Orchestration](#workflow-orchestration)
    - [Steps \& Chains](#steps--chains)
    - [Parallel Processing](#parallel-processing)
    - [Map-Reduce](#map-reduce)
    - [Branching \& Routing](#branching--routing)
    - [Stateful Workflows](#stateful-workflows)
    - [Human-in-the-Loop](#human-in-the-loop)
  - [Observability](#observability)
  - [Macros \& Developer Experience](#macros--developer-experience)
    - [Agents](#agents)
    - [Tools](#tools)
    - [Validation](#validation)
  - [Advanced Topics](#advanced-topics)
    - [Custom Adapters](#custom-adapters)
    - [Fallback Strategies](#fallback-strategies)
    - [Mocking \& Testing](#mocking--testing)
  - [Examples](#examples)
  - [Configuration](#configuration)
  - [License](#license)

---

## Overview

`gemini-structured-output` is a high-level framework built on top of `gemini-rust`. While the base client handles raw API communication, this library solves the "last mile" problem of working with LLMs in production: **reliability**.

It guarantees that the output you get from Gemini matches your Rust data structures exactly. Beyond simple schema validation, it implements a **Self-Correction Loop** using JSON Patch. If the model generates invalid data (schema violation, logic error, or context violation), the library automatically feeds the error back to the model, asks for a specific patch, and applies it transactionally.

Furthermore, it provides a **Type-Safe Workflow Engine** for building complex agentic pipelines (chains, parallel maps, reductions) with built-in metrics and tracing.

## Key Features

*   **🛡️ Type-Safe Output:** Define your output shape using standard Rust structs (`serde` + `schemars`).
*   **🔧 JSON Patch Refinement:** An autonomous engine that iteratively fixes model hallucinations or syntax errors using RFC6902 patches.
*   **⛓️ Workflow Engine:** Composable abstractions (`Step`, `Chain`, `ParallelMap`, `Reduce`) for building complex AI applications.
*   **⚡ Smart Caching:** Automatic Context Caching integration for large system prompts and schemas.
*   **🔭 Observability:** Built-in tracing, token counting, and latency metrics for every step in a workflow.
*   **🤖 Agent Macros:** Define functional agents with `#[gemini_agent]` that compile down to strongly-typed workflow steps.
*   **🛠️ Tooling System:** Ergonomic `#[gemini_tool]` macro for defining and registering tools/functions.
*   **💾 File Handling:** Seamless support for uploading PDFs, images, and videos for multimodal analysis.
*   **🔌 Adapters:** specialized serializers for `HashMap`, `Duration`, and other complex types that LLMs usually struggle with.

---

## Installation

Add the following to your `Cargo.toml`.

```toml
[dependencies]
gemini-structured-output = { version = "0.1.0", features = ["macros", "helpers"] }
serde = { version = "1.0", features = ["derive"] }
schemars = "0.8"
tokio = { version = "1", features = ["full"] }
```

> **Note:** You must set the `GEMINI_API_KEY` environment variable to use the client.

---

## Quick Start

Here is the "Hello World" of structured output. We define a struct, and the library ensures Gemini fills it out.

```rust
use gemini_structured_output::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// 1. Define your output structure
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SentimentReport {
    sentiment: String,
    score: f64,
    key_topics: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 2. Initialize the client
    let api_key = std::env::var("GEMINI_API_KEY").expect("API Key needed");
    let client = StructuredClientBuilder::new(api_key).build()?;

    // 3. Generate
    let report: SentimentReport = client
        .request::<SentimentReport>()
        .system("You are a sentiment analysis engine.")
        .user_text("I absolutely loved the new features, but the UI is a bit clunky.")
        .execute()
        .await?
        .value;

    println!("{:#?}", report);
    // Output:
    // SentimentReport {
    //     sentiment: "Mixed",
    //     score: 0.7,
    //     key_topics: ["features", "UI", "usability"]
    // }

    Ok(())
}
```

---

## Core Concepts

### Structured Generation

The library leverages `schemars` to generate an OpenAPI 3.0 schema from your Rust structs. This schema is passed to Gemini either via generation configuration (Gemini 1.5 Pro/Flash) or injected into the system prompt for legacy compatibility.

The `StructuredRequest` builder allows you to configure:

* **System Instructions:** Role definitions.
* **Temperature:** Creativity control.
* **Retries:** Network-level retry logic (429/503 handling).
* **Parse Attempts:** How many times to retry if the model outputs invalid JSON.

### The Refinement Engine

This is the crown jewel of the library. It allows you to "edit" structured data using natural language instructions, ensuring the result remains valid according to your schema.

**How it works:**

1. Takes the current struct `T`.
2. Takes a natural language `instruction` (e.g., "Change the currency to USD").
3. Sends both to Gemini, asking for a **JSON Patch**.
4. Applies the patch.
5. **Validates** the result against the schema and any custom logic.
6. If validation fails, it feeds the specific error back to Gemini and repeats the loop.

```rust
let initial_profile = Profile { name: "John", role: "Intern" };

let refined = client
    .refine(initial_profile, "Promote John to Senior Dev")
    .with_validator(|p| {
        if p.role == "Intern" { Some("Promotion didn't happen!".to_string()) } else { None }
    })
    .execute()
    .await?;

println!("Attempts: {}", refined.attempts.len());
```

### Context & Files

Gemini is multimodal. This library makes it trivial to attach files (PDFs, Images, CSVs) to your requests.

```rust
let handle = client.file_manager.upload_path("invoice.pdf").await?;

let invoice: InvoiceData = client
    .request::<InvoiceData>()
    .user_file("Extract data from this invoice", &handle)?
    .execute()
    .await?
    .value;
```

### Caching

For heavy workloads involving massive system prompts or huge schemas, you can enable Context Caching.

```rust
let client = StructuredClientBuilder::new(key)
    .with_cache_policy(CachePolicy::Enabled { 
        ttl: Duration::from_secs(600) 
    })
    .build()?;
```

The library automatically hashes your system prompt + schema + tools to create a deterministic cache key. Subsequent requests reuse the cached context, saving tokens and reducing latency.

---

## Workflow Orchestration

Building complex agents requires more than just one call. The `workflow` module provides a strongly-typed pipeline system.

### Steps & Chains

A `Step` is the fundamental unit. It takes `Input` and returns `Result<Output>`. Steps can be chained fluently.

```rust
// Chain: Input -> Step A -> Step B -> Output
let pipeline = summarizer
    .then(translator)
    .then(email_drafter);

let result = pipeline.run(input_text, &ctx).await?;
```

### Parallel Processing

Use `ParallelMapStep` to process a list of items concurrently.

```rust
// Process up to 5 documents at once
let batch_processor = ParallelMapStep::new(document_analyzer, 5);

let results: Vec<Analysis> = batch_processor.run(documents, &ctx).await?;
```

### Map-Reduce

Combine `ParallelMapStep` with `ReduceStep` to synthesize data.

```rust
let map_reduce = ParallelMapStep::new(analyzer, 4)
    .then(ReduceStep::<Analysis, Summary>::new(
        client.clone(),
        "Summarize these analyses into a final report"
    ));
```

### Branching & Routing

Use `RouterStep` to let the model decide the execution path.

```rust
let router = RouterStep::new(
    client,
    "Is this a technical query or a billing query?",
    |decision: QueryType| match decision {
        QueryType::Technical => Box::new(tech_support_agent),
        QueryType::Billing => Box::new(billing_agent),
    }
);
```

### Stateful Workflows

Avoid "tuple hell" (passing `(A, B, C)` between steps) by using `StateWorkflow`. This allows steps to read/write to a shared Blackboard struct.

```rust
#[derive(Default)]
struct AppState { doc: String, summary: Option<String>, score: f32 }

let workflow = StateWorkflow::new(AppState::default())
    .with_adapter(summarizer, |s| s.doc.clone(), |s, out| s.summary = Some(out));
```

### Human-in-the-Loop

Workflows can be paused for human review using `CheckpointStep`.

```rust
let pipeline = generator
    .then(CheckpointStep::new("ReviewDraft")) // <--- Returns Error::Checkpoint here
    .then(publisher);
```

When the checkpoint is hit, the workflow returns a special error containing the serialized intermediate state. You can save this, present it to a user UI, and resume the workflow later.

---

## Observability

The library tracks detailed metrics for every workflow execution via `ExecutionContext`.

```rust
let ctx = ExecutionContext::new();
let result = pipeline.run(input, &ctx).await?;

let metrics = ctx.snapshot();
println!("Total Tokens: {}", metrics.total_token_count);
println!("Steps Completed: {}", metrics.steps_completed);
println!("Network Retries: {}", metrics.network_attempts);
```

**Tracing:**
The context also records a structured event log (`TraceEntry`) containing:
* Step Start/End times.
* Inputs and Outputs (Artifacts).
* Errors.

---

## Macros & Developer Experience

We provide procedural macros to reduce boilerplate.

### Agents

Define a struct as an Agent. The macro implements the `Step` trait for you.

```rust
#[gemini_agent(
    input = "Article",
    output = "Summary",
    system = "You are an expert summarizer. Be concise."
)]
struct Summarizer;

// Usage
let agent = Summarizer::new(client);
```

### Tools

Turn async functions into Gemini-compatible tools.

```rust
#[gemini_tool(description = "Get current stock price")]
async fn get_stock_price(args: StockRequest) -> Result<StockPrice, ToolError> {
    // ... implementation
}

// Registering
let registry = ToolRegistry::new()
    .register_tool(get_stock_price_tool::registrar());
```

### Validation

Add runtime validation logic directly to your data structures.

```rust
#[derive(GeminiValidated, JsonSchema, ...)]
struct UserProfile {
    #[gemini(min_len = 3, max_len = 50)]
    username: String,

    #[gemini(validate_with = "validate_email")]
    email: String,

    #[gemini(min = 18, error_message = "User must be an adult")]
    age: i32,
}
```

---

## Advanced Topics

### Custom Adapters

Sometimes LLMs struggle with Rust-specific types like `HashMap` or `Duration`. We provide adapters to serialize these into LLM-friendly formats (e.g., list of Key-Value pairs).

```rust
#[derive(Serialize, Deserialize, JsonSchema)]
struct Config {
    // Serializes as [{key: "...", value: ...}] for the LLM
    // Deserializes back into HashMap for Rust
    #[serde(with = "gemini_structured_output::adapter::map")]
    #[schemars(with = "Vec<KeyValue<String, f64>>")]
    pub settings: HashMap<String, f64>,
}
```

### Fallback Strategies

Configure the client to automatically escalate to a smarter model (e.g., Pro) if the faster model (e.g., Flash) fails to produce valid output after $N$ attempts.

```rust
let client = StructuredClientBuilder::new(key)
    .with_model(Model::Gemini25Flash)
    .with_fallback_strategy(FallbackStrategy::Escalate {
        after_attempts: 2,
        target: Model::Gemini25Pro,
    })
    .build()?;
```

### Mocking & Testing

The client supports a mock handler for unit tests, allowing you to bypass the API entirely.

```rust
let client = StructuredClientBuilder::new("mock-key")
    .with_mock(|req| {
        Ok(r#"{ "name": "Mock Response" }"#.to_string())
    })
    .build()?;
```

---

## Examples

The `examples/` directory contains rich, runnable scenarios:

| Example                    | Description                                               |
| :------------------------- | :-------------------------------------------------------- |
| `basic_structured.rs`      | Simple extraction of a struct from text.                  |
| `financial_forecast.rs`    | Complex nested schemas, HashMaps, and Refinement loops.   |
| `agentic_workflow.rs`      | Branching, parallel processing, and typed steps.          |
| `tool_loop.rs`             | Using tools/function calling within a structured request. |
| `refinement_with_files.rs` | Uploading a PDF and refining extracted data against it.   |
| `observability.rs`         | Demonstrates traces, metrics, and checkpoints.            |
| `macro_tools.rs`           | Using the `#[gemini_tool]` and validation macros.         |
| `interactive_session.rs`   | A conversational loop with state management.              |

Run an example using:

```bash
GEMINI_API_KEY=your_key cargo run --example agentic_workflow --features macros
```

---

## Configuration

The `StructuredClientBuilder` offers extensive customization:

```rust
StructuredClientBuilder::new(api_key)
    .with_model(Model::Gemini25Flash)
    // Caching
    .with_cache_policy(CachePolicy::Enabled { ttl: Duration::from_secs(300) })
    // Refinement Logic
    .with_refinement_retries(3)
    .with_refinement_temperature(0.2)
    .with_refinement_strategy(PatchStrategy::PartialApply) // vs Atomic
    // Array Handling in Patches
    .with_array_strategy(ArrayPatchStrategy::ReplaceWhole)
    // Global Defaults
    .with_default_temperature(0.1)
    .with_default_retries(5)
    .build()?;
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with 🦀 Rust and Google Gemini</sub>
</div>
