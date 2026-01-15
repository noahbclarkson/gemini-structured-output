# Fix for Gemini Null Field Issue with Flattened Enums

## Problem

When using Gemini Strict Mode with complex Rust enums (particularly externally-tagged enums), Gemini was outputting responses with explicit `null` values for all non-selected enum variants:

```json
{
  "processor": {
    "model": null,
    "calculation": { "steps": [...] },
    "taxCalculation": null,
    "pnlScheduledTransactions": null,
    ...
  }
}
```

This caused serde deserialization to fail with errors like:
```
JSON parsing failed error=invalid type: null, expected internally tagged enum ForecastModel
```

## Root Cause

The issue stems from the "Impedance Mismatch" between:

1. **Gemini's Strict JSON Schema Requirements**: Gemini Strict Mode flattens `oneOf`/`anyOf` variants into a single "super-object" where all fields are optional/nullable
2. **Rust's Serde Enum Deserialization**: Serde expects exclusive variants and fails when it sees `null` values for fields that are not `Option<T>`

When Gemini sees a flattened schema, it outputs `"field": null` for variants it didn't choose, rather than omitting the key entirely.

## Solution

Added a `prune_null_fields` function that recursively removes all object keys with `null` values before serde deserialization.

### Changes Made

1. **Added `prune_null_fields` function** in `src/schema.rs`:
   - Recursively traverses JSON values
   - Removes all object keys where the value is `null`
   - Allows serde to correctly identify the active enum variant

2. **Updated normalization pipeline** in `src/request.rs`:
   - Added `prune_null_fields` call after `normalize_json_response`
   - Applied to both regular execution and streaming modes

3. **Added comprehensive tests** in `tests/repro_real_xero.rs`:
   - Unit test for basic null pruning functionality
   - Integration test with actual xero-forecasting types
   - Test with actual Gemini API call (requires API key)

## Testing

### Run Unit Tests
```bash
cd gemini-structured-output
cargo test test_prune_null_fields_basic
cargo test test_prune_nulls_from_gemini_flattened_enum
```

### Run Integration Test with Gemini
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Run the test (it's ignored by default)
cargo test test_real_xero_forecast_config_with_gemini -- --ignored
```

This test:
- Sends a real request to Gemini with the `FullForecastConfig` schema
- Asks for specific P&L account configurations
- Uses retry count of 0 for fast iteration
- Logs all errors immediately for debugging

## Technical Details

### Processing Pipeline

The JSON response now goes through these steps:

1. **Parse to Value**: `serde_json::from_str::<Value>(&text)`
2. **Normalize Maps**: `normalize_json_response()` - Converts `Array<{__key__, __value__}>` to `Object`
3. **Prune Nulls**: `prune_null_fields()` - **NEW** - Removes all null fields
4. **Recover Enums**: `recover_internally_tagged_enums()` - Expands collapsed enum strings
5. **Deserialize**: `serde_json::from_value::<T>(json_value)`

### Example Transformation

**Before pruning:**
```json
{
  "processor": {
    "model": "Auto",
    "calculation": null,
    "taxCalculation": null,
    "pnlScheduledTransactions": null,
    "targetSeek": null,
    "conditionalCalculation": null
  }
}
```

**After pruning:**
```json
{
  "processor": {
    "model": "Auto"
  }
}
```

**Serde deserialization:** ✅ Success! Identifies this as the `Model` variant.

## Performance Impact

Minimal - the function does a single recursive pass over the JSON tree, only removing entries where `v.is_null()` returns true.

## Future Considerations

If issues persist, we may need to:
1. Investigate alternative schema flattening strategies
2. Consider using a different serde tagging format (`#[serde(untagged)]`)
3. Explore modifying the schema generation to better align with Gemini's expectations

However, the current fix should resolve the immediate issue without requiring changes to the xero-forecasting codebase.
