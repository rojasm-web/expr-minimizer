// src/bin/plot_eml_from_catalog.rs
//! CLI: plot EML-derived f(x) from a catalog id.
//!
//! Usage examples:
//!   # default: uses symbolic_catalog.txt and the highest id found there
//!   cargo run --bin plot_eml_from_catalog
//!
//!   # explicit catalog and id
//!   cargo run --bin plot_eml_from_catalog -- --id 52 --catalog constants_catalog.txt --out id52_plot.png
//!
//! The evaluator implements the repo semantics: eml(a,b) = exp(a) - ln(b).
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::error::Error;

use num_complex::Complex64;
use regex::Regex;

use plotters::prelude::*;

/// Node in the parsed EML tree
#[derive(Debug, Clone)]
enum Node {
    Var,                     // x
    Const(Complex64),        // complex constant
    Prim(Box<Node>, Box<Node>), // eml(left, right)
}

/// Read the catalog and extract (symbolic_value, eml_subtree) for a given id.
fn load_catalog_entry(catalog_path: &str, target_id: usize) -> Result<(String, String), Box<dyn Error>> {
    let mut text = String::new();
    File::open(catalog_path)?.read_to_string(&mut text)?;

    let id_pattern = Regex::new(&format!(r"(?m)^id\s+{}:\s*$", target_id))?;
    let id_match = id_pattern.find(&text)
        .ok_or_else(|| format!("id {} not found in catalog {}", target_id, catalog_path))?;
    let start = id_match.end();
    let rest = &text[start..];
    let next_id_re = Regex::new(r"(?m)^\s*id\s+\d+:\s*$")?;
    let end = match next_id_re.find(rest) {
        Some(m) => m.start(),
        None => rest.len(),
    };
    let block = &rest[..end];

    let sym_re = Regex::new(r"(?m)^\s*Symbolic Value\s*:\s*(?P<val>.+)\s*$")?;
    let eml_re = Regex::new(r"(?m)^\s*EML Subtree\s*:\s*(?P<val>.+)\s*$")?;

    let sym = if let Some(c) = sym_re.captures(block) {
        c.name("val").unwrap().as_str().trim().to_string()
    } else {
        if let Some(line) = block.lines().find(|l| l.contains("Symbolic Value")) {
            line.splitn(2, ':').nth(1).unwrap_or("").trim().to_string()
        } else {
            "<unknown>".to_string()
        }
    };

    let eml = if let Some(c) = eml_re.captures(block) {
        c.name("val").unwrap().as_str().trim().to_string()
    } else {
        if let Some(line) = block.lines().find(|l| l.contains("EML Subtree")) {
            line.splitn(2, ':').nth(1).unwrap_or("").trim().to_string()
        } else {
            return Err(format!("EML Subtree not found for id {} in {}", target_id, catalog_path).into());
        }
    };

    Ok((sym, eml))
}

/// Find the maximum id present in the catalog file by scanning lines like "id <number>:".
fn find_max_id_in_catalog(catalog_path: &str) -> Result<usize, Box<dyn Error>> {
    let mut text = String::new();
    File::open(catalog_path)?.read_to_string(&mut text)?;
    let id_re = Regex::new(r"(?m)^\s*id\s+(\d+):\s*$")?;
    let mut max_id: Option<usize> = None;
    for cap in id_re.captures_iter(&text) {
        if let Some(m) = cap.get(1) {
            let v: usize = m.as_str().parse()?;
            max_id = Some(max_id.map_or(v, |cur| cur.max(v)));
        }
    }
    max_id.ok_or_else(|| format!("No ids found in catalog {}", catalog_path).into())
}

/// Very small tokenizer & parser for EML strings supporting:
/// - eml(left,right)
/// - identifiers: x, i, e, pi
/// - numeric tokens like: 3.084795, -1, 3.14i, 1.234 + -3.14159i
fn parse_eml(eml: &str) -> Result<Node, Box<dyn Error>> {
    let s = eml.trim();
    let mut pos = 0usize;
    let chars: Vec<char> = s.chars().collect();

    fn skip_ws(chars: &[char], pos: &mut usize) {
        while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }
    }

    fn match_ident(chars: &[char], pos: &mut usize) -> Option<String> {
        skip_ws(chars, pos);
        let start = *pos;
        while *pos < chars.len() && (chars[*pos].is_alphanumeric() || chars[*pos] == '_' ) {
            *pos += 1;
        }
        if start < *pos {
            Some(chars[start..*pos].iter().collect::<String>())
        } else {
            None
        }
    }

    fn peek_char(chars: &[char], pos: usize) -> Option<char> {
        chars.get(pos).cloned()
    }

    fn consume_char(chars: &[char], pos: &mut usize, expected: char) -> Result<(), Box<dyn Error>> {
        skip_ws(chars, pos);
        if let Some(c) = peek_char(chars, *pos) {
            if c == expected {
                *pos += 1;
                Ok(())
            } else {
                Err(format!("Expected '{}' but found '{}' at pos {}", expected, c, pos).into())
            }
        } else {
            Err(format!("Expected '{}' but found end of input", expected).into())
        }
    }

    fn parse_number_token(chars: &[char], pos: &mut usize) -> Result<Complex64, Box<dyn Error>> {
        skip_ws(chars, pos);
        let start = *pos;
        while *pos < chars.len() {
            let c = chars[*pos];
            if c.is_digit(10) || c == '.' || c == '+' || c == '-' || c == 'i' || c == 'I' || c == 'e' || c == 'E' || c == 'n' || c == 'N' || c.is_whitespace() {
                *pos += 1;
            } else {
                break;
            }
        }
        let raw = chars[start..*pos].iter().collect::<String>().trim().to_string();
        if raw.is_empty() {
            return Err(format!("Expected numeric token at pos {}", start).into());
        }
        parse_number_string(&raw)
    }

    fn parse_number_string(s: &str) -> Result<Complex64, Box<dyn Error>> {
        let s_trim = s.trim();
        let lower = s_trim.to_ascii_lowercase();
        if lower == "inf" || lower == "+inf" {
            return Ok(Complex64::new(f64::INFINITY, 0.0));
        }
        if lower == "-inf" {
            return Ok(Complex64::new(f64::NEG_INFINITY, 0.0));
        }

        let re_complex = Regex::new(r"(?xi)^\s*(?P<re>[+\-]?\d+(?:\.\d+)?)\s*\+\s*(?P<im>[+\-]?\d+(?:\.\d+)?)i\s*$")?;
        if let Some(caps) = re_complex.captures(s_trim) {
            let re: f64 = caps.name("re").unwrap().as_str().parse()?;
            let im: f64 = caps.name("im").unwrap().as_str().parse()?;
            return Ok(Complex64::new(re, im));
        }
        let re_im = Regex::new(r"(?xi)^\s*(?P<im>[+\-]?\d+(?:\.\d+)?)i\s*$")?;
        if let Some(caps) = re_im.captures(s_trim) {
            let im: f64 = caps.name("im").unwrap().as_str().parse()?;
            return Ok(Complex64::new(0.0, im));
        }
        if let Ok(r) = s_trim.parse::<f64>() {
            return Ok(Complex64::new(r, 0.0));
        }
        Err(format!("Unrecognized numeric token: '{}'", s_trim).into())
    }

    // Updated parse_expr_inner: try number-first when next char looks numeric
    fn parse_expr_inner(chars: &[char], pos: &mut usize) -> Result<Node, Box<dyn Error>> {
        skip_ws(chars, pos);

        // If the next char looks like the start of a number, parse a numeric token first.
        if let Some(c) = peek_char(chars, *pos) {
            if c.is_ascii_digit() || c == '+' || c == '-' {
                let num = parse_number_token(chars, pos)?;
                return Ok(Node::Const(num));
            }
        }

        // Otherwise, try identifier (eml, x, e, i, pi) or fall back to numeric (defensive)
        if let Some(ident) = match_ident(chars, pos) {
            let lower = ident.to_ascii_lowercase();
            if lower == "eml" {
                consume_char(chars, pos, '(')?;
                let left = parse_expr_inner(chars, pos)?;
                consume_char(chars, pos, ',')?;
                let right = parse_expr_inner(chars, pos)?;
                consume_char(chars, pos, ')')?;
                return Ok(Node::Prim(Box::new(left), Box::new(right)));
            } else {
                match lower.as_str() {
                    "x" => Ok(Node::Var),
                    "i"  => Ok(Node::Const(Complex64::new(0.0, 1.0))),
                    "e"  => Ok(Node::Const(Complex64::new(std::f64::consts::E, 0.0))),
                    "pi" => Ok(Node::Const(Complex64::new(std::f64::consts::PI, 0.0))),
                    other => Err(format!("Unknown identifier '{}'", other).into())
                }
            }
        } else {
            // Defensive fallback: if we didn't get an identifier, try parsing a number token.
            let num = parse_number_token(chars, pos)?;
            Ok(Node::Const(num))
        }
    }

    let node = parse_expr_inner(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos < chars.len() {
        let trailing: String = chars[pos..].iter().collect();
        if trailing.trim().is_empty() {
            Ok(node)
        } else {
            if trailing.trim_start().starts_with('[') {
                Ok(node)
            } else {
                Err(format!("Unexpected trailing content after parse: '{}'", trailing).into())
            }
        }
    } else {
        Ok(node)
    }
}

/// Evaluate a parsed Node at a real positive x, using Complex64 arithmetic.
fn eval_node(node: &Node, x: f64) -> Complex64 {
    match node {
        Node::Var => Complex64::new(x, 0.0),
        Node::Const(c) => *c,
        Node::Prim(left, right) => {
            let a = eval_node(left.as_ref(), x);
            let b = eval_node(right.as_ref(), x);
            a.exp() - b.ln()
        }
    }
}

/// Compute simple EML tree statistics: total nodes, leaves, max depth.
fn eml_stats(node: &Node) -> (usize, usize, usize) {
    fn rec(n: &Node) -> (usize, usize, usize) {
        match n {
            Node::Var | Node::Const(_) => (1usize, 1usize, 1usize),
            Node::Prim(left, right) => {
                let (n1, l1, d1) = rec(left.as_ref());
                let (n2, l2, d2) = rec(right.as_ref());
                let nodes = 1 + n1 + n2;
                let leaves = l1 + l2;
                let depth = 1 + std::cmp::max(d1, d2);
                (nodes, leaves, depth)
            }
        }
    }
    rec(node)
}

/// Helper: create n samples from xmin to xmax (inclusive)
fn linspace(xmin: f64, xmax: f64, n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    if n == 1 { return vec![xmin]; }
    let step = (xmax - xmin) / ((n - 1) as f64);
    (0..n).map(|i| xmin + (i as f64) * step).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut id_opt: Option<usize> = None;
    // default catalog is symbolic_catalog.txt per your request
    let mut catalog = "symbolic_catalog.txt".to_string();
    let mut xmin = -10.0f64;
    let mut xmax = 10.0f64;
    let mut n = 1000usize;
    let mut out = "plot.png".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--id" => { if let Some(v) = args.next() { id_opt = Some(v.parse()?); } }
            "--catalog" => { if let Some(v) = args.next() { catalog = v; } }
            "--xmin" => { if let Some(v) = args.next() { xmin = v.parse()?; } }
            "--xmax" => { if let Some(v) = args.next() { xmax = v.parse()?; } }
            "--n" => { if let Some(v) = args.next() { n = v.parse()?; } }
            "--out" => { if let Some(v) = args.next() { out = v; } }
            "--help" | "-h" => {
                println!("Usage: --id <N> [--catalog <file>] [--xmin <f>] [--xmax <f>] [--n <samples>] [--out <file>]");
                return Ok(());
            }
            other => { eprintln!("Unknown argument: {}", other); }
        }
    }

    // If id not specified, find the highest id in the chosen catalog file
    let id = match id_opt {
        Some(v) => v,
        None => {
            eprintln!("--id not provided; scanning '{}' for highest id...", catalog);
            find_max_id_in_catalog(&catalog)?
        }
    };

    let (sym, eml) = load_catalog_entry(&catalog, id)?;
    // parse eml early so we can compute stats and put them into the PNG caption
    let node = parse_eml(&eml).map_err(|e| format!("Failed to parse EML subtree: {}", e))?;
    let (nodes, leaves, depth) = eml_stats(&node);

    // Print compact info to stdout for debugging
    println!("id {} -> Symbolic Value: '{}'", id, sym);
    println!("EML summary (stdout): nodes = {}, leaves = {}, depth = {}", nodes, leaves, depth);

    // Prepare samples and evaluate
    let xs = linspace(xmin, xmax, n);
    let mut revals: Vec<(f64, f64)> = Vec::with_capacity(xs.len());
    let mut imvals: Vec<(f64, f64)> = Vec::with_capacity(xs.len());
    for &x in &xs {
        let y = eval_node(&node, x);
        revals.push((x, y.re));
        imvals.push((x, y.im));
    }

    // Determine y range (include both real and imaginary parts)
    let mut ymin = std::f64::INFINITY;
    let mut ymax = std::f64::NEG_INFINITY;
    for &(_, v) in revals.iter().chain(imvals.iter()) {
        if v.is_finite() {
            if v < ymin { ymin = v; }
            if v > ymax { ymax = v; }
        }
    }
    if ymin == std::f64::INFINITY || ymax == std::f64::NEG_INFINITY {
        ymin = -1.0; ymax = 1.0;
    }
    if (ymax - ymin).abs() < 1e-6 {
        ymax += 0.5; ymin -= 0.5;
    }
    let yrange = ymax - ymin;
    ymin -= yrange * 0.08;
    ymax += yrange * 0.08;

    // Create drawing area and use Symbolic Value + EML summary as the PNG caption
    let out_path = PathBuf::from(&out);
    let root = BitMapBackend::new(&out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let caption_text = format!("{}   EML summary: nodes = {}, leaves = {}, depth = {}", sym, nodes, leaves, depth);

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(caption_text, ("sans-serif", 18).into_font())
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    chart.configure_mesh()
        .x_desc("x (Real Positive Domain)")
        .y_desc("f(x)")
        .draw()?;

    // Draw real part (solid)
    chart.draw_series(LineSeries::new(
        revals.iter().cloned(),
        ShapeStyle::from(&RGBColor(200, 40, 40)).stroke_width(2),
    ))?;

    // Draw imaginary part as dashed (short segments)
    let max_imag = imvals.iter().map(|&(_, v)| v.abs()).fold(0.0f64, |a,b| a.max(b));
    if max_imag > 1e-10 {
        let dash_len = 6usize;
        let gap_len = 4usize;
        let mut idx = 0usize;
        while idx < imvals.len() {
            let end = std::cmp::min(idx + dash_len, imvals.len());
            let seg: Vec<(f64, f64)> = imvals[idx..end].iter().cloned().collect();
            if seg.len() >= 2 {
                chart.draw_series(LineSeries::new(
                    seg,
                    ShapeStyle::from(&RGBColor(40, 80, 200)).stroke_width(1),
                ))?;
            }
            idx = end + gap_len;
        }
    }

    // Legend sample lines and labels (drawn into the chart area)
    let lx = xmin + (xmax - xmin) * 0.02;
    let ly = ymax - (ymax - ymin) * 0.06;
    let lx2 = lx + (xmax - xmin) * 0.08;

    chart.draw_series(LineSeries::new(
        vec![(lx, ly), (lx2, ly)],
        ShapeStyle::from(&RGBColor(200,40,40)).stroke_width(2),
    ))?;
    chart.draw_series(std::iter::once(Text::new(
        "Re f(x)",
        ((lx2 + (xmax - xmin)*0.02), ly),
        ("sans-serif", 15).into_font(),
    )))?;

    let ly2 = ly - (ymax - ymin) * 0.06;
    let legend_points = vec![
        (lx, ly2),
        (lx + (lx2 - lx) * 0.25, ly2),
        (lx + (lx2 - lx) * 0.5, ly2),
        (lx + (lx2 - lx) * 0.75, ly2),
        (lx2, ly2),
    ];
    // dashed legend: draw small segments
    let mut idx = 0usize;
    let dash_len = 2usize;
    let gap_len = 2usize;
    while idx < legend_points.len() {
        let end = std::cmp::min(idx + dash_len, legend_points.len());
        let seg: Vec<(f64, f64)> = legend_points[idx..end].iter().cloned().collect();
        if seg.len() >= 2 {
            chart.draw_series(LineSeries::new(
                seg,
                ShapeStyle::from(&RGBColor(40,80,200)).stroke_width(1),
            ))?;
        }
        idx = end + gap_len;
    }
    chart.draw_series(std::iter::once(Text::new(
        "Im f(x)",
        ((lx2 + (xmax - xmin)*0.02), ly2),
        ("sans-serif", 15).into_font(),
    )))?;

    root.present()?;
    println!("Saved plot to {}", out_path.display());
    Ok(())
}