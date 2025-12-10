//! Formatting helpers for preparing prompt inputs.
//!
//! This module provides utilities for converting various data formats
//! into LLM-friendly representations (typically markdown).
//!
//! Enable with the `helpers` feature flag.

use std::fmt::Write;

/// Convert CSV data to a markdown table.
///
/// # Example
/// ```
/// use gemini_structured_output::helpers::csv_to_markdown;
///
/// let csv = "Name,Age,City\nAlice,30,NYC\nBob,25,LA";
/// let md = csv_to_markdown(csv, None).unwrap();
/// assert!(md.contains("| Name"));
/// assert!(md.contains("| Alice"));
/// ```
pub fn csv_to_markdown(csv: &str, title: Option<&str>) -> Result<String, CsvError> {
    csv_to_markdown_with_options(csv, title, CsvOptions::default())
}

/// Options for CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Delimiter character (default: ',')
    pub delimiter: char,
    /// Whether the first row is a header (default: true)
    pub has_header: bool,
    /// Maximum number of rows to include (default: None = all)
    pub max_rows: Option<usize>,
    /// Columns to include by index (default: None = all)
    pub columns: Option<Vec<usize>>,
    /// Text alignment for columns
    pub alignment: TableAlignment,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            delimiter: ',',
            has_header: true,
            max_rows: None,
            columns: None,
            alignment: TableAlignment::Left,
        }
    }
}

/// Table column alignment.
#[derive(Debug, Clone, Copy, Default)]
pub enum TableAlignment {
    #[default]
    Left,
    Center,
    Right,
}

/// CSV parsing error.
#[derive(Debug, thiserror::Error)]
pub enum CsvError {
    #[error("Empty CSV data")]
    Empty,
    #[error("No data rows found")]
    NoData,
    #[error("Inconsistent column count: expected {expected}, found {found} at row {row}")]
    InconsistentColumns {
        expected: usize,
        found: usize,
        row: usize,
    },
}

/// Convert CSV to markdown with custom options.
pub fn csv_to_markdown_with_options(
    csv: &str,
    title: Option<&str>,
    options: CsvOptions,
) -> Result<String, CsvError> {
    let lines: Vec<&str> = csv.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return Err(CsvError::Empty);
    }

    let parse_row = |line: &str| -> Vec<String> {
        line.split(options.delimiter)
            .map(|s| s.trim().to_string())
            .collect()
    };

    let mut rows: Vec<Vec<String>> = lines.iter().map(|l| parse_row(l)).collect();

    // Filter columns if specified
    if let Some(ref cols) = options.columns {
        rows = rows
            .into_iter()
            .map(|row| cols.iter().filter_map(|&i| row.get(i).cloned()).collect())
            .collect();
    }

    // Apply max_rows limit (excluding header)
    if let Some(max) = options.max_rows {
        if options.has_header && rows.len() > max + 1 {
            rows.truncate(max + 1);
        } else if !options.has_header && rows.len() > max {
            rows.truncate(max);
        }
    }

    if rows.is_empty() {
        return Err(CsvError::Empty);
    }

    let col_count = rows[0].len();

    // Validate column consistency
    for (i, row) in rows.iter().enumerate() {
        if row.len() != col_count {
            return Err(CsvError::InconsistentColumns {
                expected: col_count,
                found: row.len(),
                row: i + 1,
            });
        }
    }

    // Calculate column widths
    let mut widths: Vec<usize> = vec![0; col_count];
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    // Build markdown
    let mut output = String::new();

    if let Some(t) = title {
        writeln!(output, "### {}\n", t).unwrap();
    }

    // Header row
    let header = if options.has_header {
        &rows[0]
    } else {
        // Generate column headers
        &(0..col_count)
            .map(|i| format!("Col{}", i + 1))
            .collect::<Vec<_>>()
    };

    write!(output, "|").unwrap();
    for (i, cell) in header.iter().enumerate() {
        write!(output, " {:width$} |", cell, width = widths[i]).unwrap();
    }
    writeln!(output).unwrap();

    // Separator row
    write!(output, "|").unwrap();
    for width in &widths {
        let sep_width = (*width).max(3);
        match options.alignment {
            TableAlignment::Left => {
                write!(output, " {:-<width$} |", "", width = sep_width).unwrap()
            }
            TableAlignment::Center => write!(
                output,
                ":{:-<width$}:|",
                "",
                width = sep_width.saturating_sub(2)
            )
            .unwrap(),
            TableAlignment::Right => write!(
                output,
                " {:-<width$}:|",
                "",
                width = sep_width.saturating_sub(1)
            )
            .unwrap(),
        }
    }
    writeln!(output).unwrap();

    // Data rows
    let data_start = if options.has_header { 1 } else { 0 };
    for row in rows.iter().skip(data_start) {
        write!(output, "|").unwrap();
        for (i, cell) in row.iter().enumerate() {
            write!(output, " {:width$} |", cell, width = widths[i]).unwrap();
        }
        writeln!(output).unwrap();
    }

    Ok(output)
}

/// Convert a JSON array to a markdown table.
///
/// Expects an array of objects with consistent keys.
pub fn json_array_to_markdown(
    json: &serde_json::Value,
    title: Option<&str>,
) -> Result<String, JsonTableError> {
    let array = json.as_array().ok_or(JsonTableError::NotArray)?;

    if array.is_empty() {
        return Err(JsonTableError::Empty);
    }

    // Get headers from first object
    let first = array[0].as_object().ok_or(JsonTableError::NotObjectArray)?;
    let headers: Vec<&String> = first.keys().collect();

    if headers.is_empty() {
        return Err(JsonTableError::Empty);
    }

    // Build rows
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(array.len() + 1);
    rows.push(headers.iter().map(|h| (*h).clone()).collect());

    for item in array {
        let obj = item.as_object().ok_or(JsonTableError::NotObjectArray)?;
        let row: Vec<String> = headers
            .iter()
            .map(|h| obj.get(*h).map(value_to_string).unwrap_or_default())
            .collect();
        rows.push(row);
    }

    // Convert to markdown
    let mut output = String::new();

    if let Some(t) = title {
        writeln!(output, "### {}\n", t).unwrap();
    }

    // Calculate widths
    let col_count = headers.len();
    let mut widths: Vec<usize> = vec![0; col_count];
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    // Header
    write!(output, "|").unwrap();
    for (i, header) in rows[0].iter().enumerate() {
        write!(output, " {:width$} |", header, width = widths[i]).unwrap();
    }
    writeln!(output).unwrap();

    // Separator
    write!(output, "|").unwrap();
    for width in &widths {
        write!(output, " {:-<width$} |", "", width = *width).unwrap();
    }
    writeln!(output).unwrap();

    // Data
    for row in rows.iter().skip(1) {
        write!(output, "|").unwrap();
        for (i, cell) in row.iter().enumerate() {
            write!(output, " {:width$} |", cell, width = widths[i]).unwrap();
        }
        writeln!(output).unwrap();
    }

    Ok(output)
}

fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        _ => v.to_string(),
    }
}

/// JSON to table conversion error.
#[derive(Debug, thiserror::Error)]
pub enum JsonTableError {
    #[error("Expected a JSON array")]
    NotArray,
    #[error("Expected an array of objects")]
    NotObjectArray,
    #[error("Empty data")]
    Empty,
}

/// Format numbers with thousands separators.
pub fn format_number(n: f64, decimals: usize) -> String {
    let formatted = format!("{:.prec$}", n, prec = decimals);
    let parts: Vec<&str> = formatted.split('.').collect();
    let integer_part = parts[0];

    let with_commas: String = integer_part
        .chars()
        .rev()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if i > 0 && i % 3 == 0 && c != '-' {
                acc.push(',');
            }
            acc.push(c);
            acc
        })
        .chars()
        .rev()
        .collect();

    if parts.len() > 1 {
        format!("{}.{}", with_commas, parts[1])
    } else {
        with_commas
    }
}

/// Format currency with symbol and thousands separators.
pub fn format_currency(amount: f64, currency: &str, decimals: usize) -> String {
    let symbol = match currency.to_uppercase().as_str() {
        "USD" => "$",
        "EUR" => "€",
        "GBP" => "£",
        "JPY" => "¥",
        "NZD" | "AUD" | "CAD" => "$",
        _ => "",
    };
    format!("{}{}", symbol, format_number(amount, decimals))
}

/// Truncate text with ellipsis.
pub fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else if max_len <= 3 {
        text.chars().take(max_len).collect()
    } else {
        format!("{}...", &text[..max_len - 3])
    }
}

/// Create a bullet list from items.
pub fn bullet_list<I, S>(items: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    items
        .into_iter()
        .map(|s| format!("- {}", s.as_ref()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create a numbered list from items.
pub fn numbered_list<I, S>(items: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    items
        .into_iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s.as_ref()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format a key-value pair for prompts.
pub fn key_value(key: &str, value: impl std::fmt::Display) -> String {
    format!("**{}**: {}", key, value)
}

/// Format multiple key-value pairs.
pub fn key_value_block<I, K, V>(pairs: I) -> String
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: std::fmt::Display,
{
    pairs
        .into_iter()
        .map(|(k, v)| key_value(k.as_ref(), v))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Wrap text in a code block with optional language.
pub fn code_block(code: &str, language: Option<&str>) -> String {
    format!("```{}\n{}\n```", language.unwrap_or(""), code)
}

/// Create a collapsible section (details/summary in markdown).
pub fn collapsible(summary: &str, content: &str) -> String {
    format!(
        "<details>\n<summary>{}</summary>\n\n{}\n</details>",
        summary, content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_to_markdown() {
        let csv = "Name,Age,City\nAlice,30,New York\nBob,25,Los Angeles";
        let md = csv_to_markdown(csv, Some("People")).unwrap();
        assert!(md.contains("### People"));
        assert!(md.contains("| Name"));
        assert!(md.contains("| Alice"));
    }

    #[test]
    fn test_csv_with_options() {
        let csv = "a;b;c\n1;2;3\n4;5;6";
        let opts = CsvOptions {
            delimiter: ';',
            ..Default::default()
        };
        let md = csv_to_markdown_with_options(csv, None, opts).unwrap();
        assert!(md.contains("| a "));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234567.89, 2), "1,234,567.89");
        assert_eq!(format_number(-1000.0, 0), "-1,000");
    }

    #[test]
    fn test_format_currency() {
        assert_eq!(format_currency(1234.56, "USD", 2), "$1,234.56");
        assert_eq!(format_currency(1000.0, "EUR", 2), "€1,000.00");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate_text("Hello World", 8), "Hello...");
        assert_eq!(truncate_text("Hi", 10), "Hi");
    }

    #[test]
    fn test_bullet_list() {
        let list = bullet_list(["Apple", "Banana"]);
        assert_eq!(list, "- Apple\n- Banana");
    }

    #[test]
    fn test_numbered_list() {
        let list = numbered_list(["First", "Second"]);
        assert_eq!(list, "1. First\n2. Second");
    }

    #[test]
    fn test_json_array_to_markdown() {
        let json = serde_json::json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]);
        let md = json_array_to_markdown(&json, None).unwrap();
        assert!(md.contains("| name"));
        assert!(md.contains("| Alice"));
    }
}
