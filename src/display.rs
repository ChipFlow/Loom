// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Display ($display/$write) support for GEM.
// This module handles parsing format strings from Yosys JSON output
// and formatting display messages at runtime.

use indexmap::IndexMap;
use std::path::Path;

/// Information about a GEM_DISPLAY cell extracted from Yosys JSON.
#[derive(Debug, Clone)]
pub struct DisplayCellInfo {
    /// Format string from the gem_format attribute.
    pub format: String,
    /// Argument width in bits from gem_args_width attribute.
    pub args_width: u32,
}

/// Extract display cell information from a Yosys JSON file.
/// Returns a map from cell name to DisplayCellInfo.
pub fn extract_display_info_from_json(
    json_path: &Path
) -> Result<IndexMap<String, DisplayCellInfo>, String> {
    let content = std::fs::read_to_string(json_path)
        .map_err(|e| format!("Failed to read JSON file: {}", e))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let mut result = IndexMap::new();

    // Navigate to modules
    let modules = json.get("modules")
        .and_then(|m| m.as_object())
        .ok_or("No modules in JSON")?;

    for (_module_name, module) in modules {
        let cells = match module.get("cells").and_then(|c| c.as_object()) {
            Some(c) => c,
            None => continue,
        };

        for (cell_name, cell) in cells {
            // Check if this is a GEM_DISPLAY cell
            let cell_type = cell.get("type").and_then(|t| t.as_str());
            if cell_type != Some("GEM_DISPLAY") {
                continue;
            }

            // Get attributes
            let attrs = match cell.get("attributes").and_then(|a| a.as_object()) {
                Some(a) => a,
                None => continue,
            };

            // Get gem_format attribute
            let format = attrs.get("gem_format")
                .and_then(|f| f.as_str())
                .unwrap_or("")
                .to_string();

            // Get gem_args_width attribute (stored as binary string)
            let args_width = attrs.get("gem_args_width")
                .and_then(|w| w.as_str())
                .and_then(|s| u32::from_str_radix(s, 2).ok())
                .unwrap_or(32);

            result.insert(cell_name.clone(), DisplayCellInfo {
                format,
                args_width,
            });
        }
    }

    Ok(result)
}

/// Parse a Yosys format string and extract format specifiers.
/// Yosys uses Python-style format: {width:format-spec}
/// Example: "{8:>02h-u}" means 8 bits, right-aligned, zero-padded, hex, unsigned
#[derive(Debug, Clone)]
pub struct FormatSpec {
    /// Bit width of the argument.
    pub width: u32,
    /// Display format: 'd' decimal, 'h' hex, 'b' binary, 's' string.
    pub radix: char,
    /// Whether the value is signed.
    pub signed: bool,
    /// Minimum display width (for padding).
    pub min_width: Option<u32>,
    /// Pad character ('0' or ' ').
    pub pad_char: char,
    /// Alignment: '<' left, '>' right, '^' center.
    pub align: char,
}

/// Parse Yosys format string into literal parts and format specifiers.
pub fn parse_format_string(format: &str) -> (Vec<String>, Vec<FormatSpec>) {
    let mut literals = Vec::new();
    let mut specs = Vec::new();
    let mut current_literal = String::new();
    let mut chars = format.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            // Start of format specifier
            literals.push(std::mem::take(&mut current_literal));

            let mut spec_str = String::new();
            while let Some(&next) = chars.peek() {
                if next == '}' {
                    chars.next();
                    break;
                }
                spec_str.push(chars.next().unwrap());
            }

            // Parse spec_str: "width:format" or "width"
            let (width_part, format_part) = spec_str
                .split_once(':')
                .map(|(w, f)| (w, Some(f)))
                .unwrap_or((&spec_str, None));

            let width: u32 = width_part.parse().unwrap_or(32);

            // Default spec values
            let mut radix = 'd';
            let mut signed = true;
            let mut min_width = None;
            let mut pad_char = ' ';
            let mut align = '>';

            if let Some(fmt) = format_part {
                // Parse format part like ">02h-u" or "d"
                let fmt_chars: Vec<char> = fmt.chars().collect();
                let mut i = 0;

                // Check for alignment
                if i < fmt_chars.len() && (fmt_chars[i] == '<' || fmt_chars[i] == '>' || fmt_chars[i] == '^') {
                    align = fmt_chars[i];
                    i += 1;
                }

                // Check for pad character
                if i < fmt_chars.len() && fmt_chars[i] == '0' {
                    pad_char = '0';
                    i += 1;
                }

                // Parse minimum width
                let mut width_str = String::new();
                while i < fmt_chars.len() && fmt_chars[i].is_ascii_digit() {
                    width_str.push(fmt_chars[i]);
                    i += 1;
                }
                if !width_str.is_empty() {
                    min_width = width_str.parse().ok();
                }

                // Parse radix (h, d, b, s)
                if i < fmt_chars.len() {
                    match fmt_chars[i] {
                        'h' | 'x' => radix = 'h',
                        'd' => radix = 'd',
                        'b' => radix = 'b',
                        's' => radix = 's',
                        _ => {}
                    }
                    i += 1;
                }

                // Check for sign indicator
                if i < fmt_chars.len() && fmt_chars[i] == '-' {
                    i += 1;
                    if i < fmt_chars.len() {
                        match fmt_chars[i] {
                            'u' => signed = false,
                            's' => signed = true,
                            _ => {}
                        }
                    }
                }
            }

            specs.push(FormatSpec {
                width,
                radix,
                signed,
                min_width,
                pad_char,
                align,
            });
        } else if c == '\\' {
            // Handle escape sequences
            if let Some(&next) = chars.peek() {
                match next {
                    'n' => { chars.next(); current_literal.push('\n'); }
                    't' => { chars.next(); current_literal.push('\t'); }
                    '\\' => { chars.next(); current_literal.push('\\'); }
                    _ => current_literal.push(c),
                }
            } else {
                current_literal.push(c);
            }
        } else {
            current_literal.push(c);
        }
    }

    literals.push(current_literal);

    (literals, specs)
}

/// Format a display message given the format string and argument values.
pub fn format_display_message(format: &str, args: &[u64], arg_widths: &[u32]) -> String {
    let (literals, specs) = parse_format_string(format);

    let mut result = String::new();
    let mut arg_idx = 0;

    for (i, literal) in literals.iter().enumerate() {
        result.push_str(literal);

        if i < specs.len() && arg_idx < args.len() {
            let spec = &specs[i];
            let value = args[arg_idx];
            let width = arg_widths.get(arg_idx).copied().unwrap_or(spec.width);

            // Mask value to specified width
            let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
            let masked_value = value & mask;

            // Handle signed values
            let display_value = if spec.signed && width < 64 {
                let sign_bit = 1u64 << (width - 1);
                if masked_value & sign_bit != 0 {
                    // Sign extend for display
                    (masked_value | !mask) as i64
                } else {
                    masked_value as i64
                }
            } else {
                masked_value as i64
            };

            // Format based on radix
            let formatted = match spec.radix {
                'h' | 'x' => format!("{:x}", masked_value),
                'b' => format!("{:b}", masked_value),
                'd' => {
                    if spec.signed {
                        format!("{}", display_value)
                    } else {
                        format!("{}", masked_value)
                    }
                }
                _ => format!("{}", masked_value),
            };

            // Apply padding and alignment
            if let Some(min_w) = spec.min_width {
                let min_w = min_w as usize;
                if formatted.len() < min_w {
                    let padding = min_w - formatted.len();
                    let pad = spec.pad_char.to_string().repeat(padding);
                    match spec.align {
                        '<' => result.push_str(&formatted),
                        '^' => {
                            let left = padding / 2;
                            let right = padding - left;
                            result.push_str(&spec.pad_char.to_string().repeat(left));
                            result.push_str(&formatted);
                            result.push_str(&spec.pad_char.to_string().repeat(right));
                        }
                        _ => {
                            result.push_str(&pad);
                            result.push_str(&formatted);
                        }
                    }
                } else {
                    result.push_str(&formatted);
                }
            } else {
                result.push_str(&formatted);
            }

            arg_idx += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_format_string() {
        let (literals, specs) = parse_format_string("Counter = {8:>02h-u}\\n");
        assert_eq!(literals, vec!["Counter = ", "\n"]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].width, 8);
        assert_eq!(specs[0].radix, 'h');
        assert!(!specs[0].signed);
    }

    #[test]
    fn test_format_display_message() {
        let msg = format_display_message(
            "value = {8:>02h-u}",
            &[255],
            &[8]
        );
        assert_eq!(msg, "value = ff");

        let msg = format_display_message(
            "Counter reached 5, data_in = {8:>02h-u}\\n",
            &[0x42],
            &[8]
        );
        assert_eq!(msg, "Counter reached 5, data_in = 42\n");
    }
}
