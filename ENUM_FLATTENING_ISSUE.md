# Critical Issue: Nested Enum Flattening Incompatibility

## The Problem

The schema flattening approach in `gemini-structured-output` has a fundamental incompatibility with complex nested Rust enums (especially externally-tagged enums).

### Example

**Rust Definition:**
```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PnlProcessor {
    Model(ForecastModel),  // Externally tagged
    Calculation { steps: Vec<CalculationStep> },
    // ... other variants
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ForecastModel {
    Auto,
    Mstl {
        seasonal_periods: Vec<usize>,
        trend_model: MstlTrendModel,
    },
    // ... other variants
}
```

**Expected JSON (for `PnlProcessor::Model(ForecastModel::Mstl{...})`):**
```json
{
  "model": {
    "Mstl": {
      "seasonalPeriods": [12],
      "trendModel": "Ets"
    }
  }
}
```

**What Gemini Outputs:**
```json
{
  "model": "mstl",
  "seasonalPeriods": [12],
  "trendModel": "ets"
}
```

**Serde Error:**
```
invalid value: map, expected map with a single key
```

## Root Cause Analysis

### 1. JSON Schema Generation
The schema generator creates an `anyOf` structure for enums:
```json
{
  "anyOf": [
    {
      "type": "object",
      "properties": {
        "model": {
          "anyOf": [
            { "type": "string", "enum": ["Auto"] },
            {
              "type": "object",
              "properties": {
                "Mstl": {
                  "type": "object",
                  "properties": {
                    "seasonalPeriods": { "type": "array", "items": { "type": "integer" } },
                    "trendModel": { "type": "string", "enum": ["Naive", "Ets"] }
                  }
                }
              }
            }
          ]
        }
      },
      "required": ["model"]
    },
    ...other variants...
  ]
}
```

### 2. Schema Flattening for Gemini Strict Mode
When flattened for Gemini Strict Mode (which doesn't handle `oneOf`/`anyOf` well), all fields from all variants are merged into a single object:
```json
{
  "type": "object",
  "properties": {
    "model": { "type": "string" },
    "seasonalPeriods": { "type": "array" },
    "trendModel": { "type": "string" },
    "calculation": { "type": "object" },
    "taxCalculation": { "type": "object" },
    // ... all other variant fields ...
  }
}
```

### 3. Gemini's Interpretation
Gemini sees this as a single object where multiple fields can coexist. It outputs:
- The discriminator field (`model`: "mstl")
- The fields for that variant (`seasonalPeriods`, `trendModel`)
- All in the same flat object

### 4. Serde Deserialization Fails
Serde's externally-tagged enum deserializer expects:
- Exactly ONE key in the object (the variant name)
- The value of that key contains the variant's data

When it sees multiple keys (`model`, `seasonalPeriods`, `trendModel`), it rejects it with "expected map with a single key".

## Additional Issues Observed

### Array Format for Complex Types
Gemini is also outputting tuple-like arrays for struct types:

**Gemini Output:**
```json
"taxCalculation": ["Profit Before Tax", 0.28]
"daysOutstanding": ["Sales", "Other Income"]
"scheduledTransaction": [-10000, "2025-08-31", 1]
```

**Expected:**
```json
"taxCalculation": {
  "profitBase": [...],
  "taxRate": 0.28
}
"daysOutstanding": {
  "Auto": {
    "drivers": [{"accountValue": "Sales"}, ...],
    "lags": [[...]]
  }
}
```

This suggests Gemini is interpreting the flattened schema as allowing positional arguments or is hallucinating a simplified format.

## Potential Solutions

### Solution 1: Enhanced Recovery Logic ⚠️ (Complex)
Add post-processing to detect flattened enums and restructure them:

```rust
pub fn unflatten_externally_tagged_enums(value: &mut Value, schema: &Value) {
    // 1. Parse schema to identify enum variants and their fields
    // 2. Detect when multiple variant fields are present in same object
    // 3. Identify correct variant based on discriminator or field presence
    // 4. Restructure: {"model": "mstl", "seasonalPeriods": [...]}
    //    -> {"model": {"Mstl": {"seasonalPeriods": [...]}}}
}
```

**Pros:**
- Minimal changes to existing schema generation
- Works with current Gemini Strict Mode

**Cons:**
- Very complex to implement correctly
- Requires deep schema understanding at runtime
- Fragile - easy to break with schema changes
- Doesn't solve the array format issue

### Solution 2: Different Serde Representation 🔨 (Requires xero-forecasting changes)
Change enum tagging to internally-tagged or untagged:

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]  // Internally tagged
pub enum PnlProcessor {
    #[serde(rename = "model")]
    Model {
        #[serde(flatten)]
        model: ForecastModel
    },
    ...
}
```

**Pros:**
- More compatible with flattened schemas
- Cleaner JSON structure

**Cons:**
- Requires changes to xero-forecasting
- May affect existing serialized data
- Doesn't fully solve nested enum problem

### Solution 3: Disable Schema Flattening for Enums 🎯 (Recommended Short-term)
Modify schema generation to preserve `anyOf` for complex enums and let Gemini handle them:

```rust
// In schema flattening logic, detect complex nested enums and preserve structure
if is_complex_nested_enum(&schema) {
    return schema; // Don't flatten
}
```

**Pros:**
- Minimal changes
- Preserves semantic structure
- May work if Gemini is smart enough to handle simple `anyOf`

**Cons:**
- May not work with Gemini Strict Mode requirements
- Need to test if Gemini can handle this

### Solution 4: Use Response Schema Mode Instead of Strict Mode 🚀 (Recommended Long-term)
Don't use Gemini's `responseMimeType: application/json` with `response_schema`. Instead:
- Send schema as system prompt guidance
- Use regular JSON parsing
- Rely on retry logic for malformed responses

**Pros:**
- No schema flattening needed
- Gemini has more flexibility in output format
- May produce better results for complex types

**Cons:**
- Potential for more parsing failures
- Need more retries
- Less guaranteed structure

## Immediate Workaround

For now, you could simplify the xero-forecasting config structure by:

1. Using simpler, flatter enum structures
2. Avoiding deeply nested enums
3. Using string discriminators with separate config objects

**Example:**
```rust
pub struct PnlAccountOverride {
    pub processor_type: ProcessorType,  // Simple enum: "model" | "calculation" | ...
    pub model_config: Option<ForecastModelConfig>,
    pub calculation_config: Option<CalculationConfig>,
    // ...
}
```

## Recommendation

I recommend a **two-phase approach**:

**Phase 1 (Immediate):** Implement Solution 3
- Modify schema flattening to preserve enum structure for complex nested enums
- Test with Gemini to see if it can handle the `anyOf`

**Phase 2 (If Phase 1 fails):** Implement Solution 1
- Build sophisticated recovery logic to unflatten nested enums
- Handle the array format issue for complex structs

**Phase 3 (Long-term):** Consider Solution 4
- Evaluate whether Strict Mode is necessary
- Test non-strict mode with schema-guided generation

## Testing

Run the actual Gemini test to see how it performs:
```bash
export GEMINI_API_KEY="your-key"
cd gemini-structured-output
cargo test test_real_xero_forecast_config_with_gemini -- --ignored --nocapture
```

This will show exactly what Gemini outputs with the current schema and help determine the best path forward.
