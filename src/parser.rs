//! A small, dependency-free recursive-descent parser so expressions can be
//! written as text instead of chained `Interner::intern_*` calls.
//!
//! Supported grammar (whitespace-insensitive):
//!
//! ```text
//! expr    := "x"
//!          | "i"                      (imaginary unit, 0+1i)
//!          | "pi"                     (real constant, std::f64::consts::PI)
//!          | "e"                      (real constant, std::f64::consts::E)
//!          | number
//!          | ("f" | "eml") "(" expr "," expr ")"
//!
//! number  := ["-"] digits ["." digits] [("+"|"-") digits ["." digits] "i"]
//!          | ["-"] digits ["." digits] "i"
//! ```
//!
//! Examples: `x`, `1`, `-2.5`, `1+2i`, `3-0.5i`, `f(x, 1)`, `f(f(x,1), i)`,
//! `eml(pi, f(x, 1))`.
//!
//! This is intentionally forgiving about the outer function name (`f` or
//! `eml` both mean the single binary primitive) since both spellings show
//! up in discussion of this codebase.

use num_complex::Complex64;

use crate::arena::Interner;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub pos: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at byte {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse `src` and intern the resulting expression into `interner`,
/// returning the root node id.
pub fn parse(interner: &mut Interner, src: &str) -> Result<u32, ParseError> {
    let chars: Vec<char> = src.chars().collect();
    let mut p = Parser { chars: &chars, pos: 0, interner };
    p.skip_ws();
    let id = p.parse_expr()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(p.err("trailing input after a complete expression"));
    }
    Ok(id)
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
    interner: &'a mut Interner,
}

impl<'a> Parser<'a> {
    fn err(&self, message: &str) -> ParseError {
        ParseError { message: message.to_string(), pos: self.pos }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: char) -> Result<(), ParseError> {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{c}'")))
        }
    }

    fn parse_expr(&mut self) -> Result<u32, ParseError> {
        self.skip_ws();
        match self.peek() {
            None => Err(self.err("unexpected end of input")),
            Some(c) if c.is_ascii_digit() || c == '-' || c == '.' => self.parse_number(),
            Some(c) if c.is_alphabetic() => self.parse_ident_expr(),
            Some(c) => Err(self.err(&format!("unexpected character '{c}'"))),
        }
    }

    fn parse_ident_expr(&mut self) -> Result<u32, ParseError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            self.pos += 1;
        }
        let ident: String = self.chars[start..self.pos].iter().collect();
        self.skip_ws();

        match ident.as_str() {
            "x" | "X" => Ok(self.interner.intern_var()),
            "i" | "I" => Ok(self.interner.intern_const(Complex64::new(0.0, 1.0))),
            "pi" | "PI" | "Pi" => {
                Ok(self.interner.intern_const(Complex64::new(std::f64::consts::PI, 0.0)))
            }
            "e" | "E" => Ok(self.interner.intern_const(Complex64::new(std::f64::consts::E, 0.0))),
            "f" | "eml" | "F" | "EML" => self.parse_prim_call(),
            other => Err(ParseError {
                message: format!("unknown identifier '{other}' (expected x, i, pi, e, or f(...)/eml(...))"),
                pos: start,
            }),
        }
    }

    fn parse_prim_call(&mut self) -> Result<u32, ParseError> {
        self.expect('(')?;
        let a = self.parse_expr()?;
        self.expect(',')?;
        let b = self.parse_expr()?;
        self.expect(')')?;
        Ok(self.interner.intern_prim(a, b))
    }

    /// Parses real, imaginary, or combined `re+imi` / `re-imi` literals.
    fn parse_number(&mut self) -> Result<u32, ParseError> {
        let (first, first_is_imag) = self.parse_signed_float_maybe_i()?;

        self.skip_ws();
        // Optional second term: (+|-) unsigned_float "i" -- only valid if
        // the first term wasn't already imaginary.
        if !first_is_imag && matches!(self.peek(), Some('+') | Some('-')) {
            let sign = if self.peek() == Some('-') { -1.0 } else { 1.0 };
            self.pos += 1;
            self.skip_ws();
            let (mag, is_imag) = self.parse_signed_float_maybe_i()?;
            if !is_imag {
                return Err(self.err("expected 'i' suffix on the second term of a complex literal"));
            }
            let v = Complex64::new(first, sign * mag);
            return Ok(self.interner.intern_const(v));
        }

        let v = if first_is_imag { Complex64::new(0.0, first) } else { Complex64::new(first, 0.0) };
        Ok(self.interner.intern_const(v))
    }

    /// Parses `["-"] digits ["." digits]`, optionally followed directly by
    /// an `i` suffix. Returns `(magnitude, was_imaginary)`.
    fn parse_signed_float_maybe_i(&mut self) -> Result<(f64, bool), ParseError> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some('.') {
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if self.pos == digits_start {
            return Err(self.err("expected a number"));
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let value: f64 = text.parse().map_err(|_| self.err("invalid numeric literal"))?;

        let is_imag = if matches!(self.peek(), Some('i') | Some('I')) {
            self.pos += 1;
            true
        } else {
            false
        };
        Ok((value, is_imag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Interner, Op};
    use crate::fingerprint::sample_bytes;

    #[test]
    fn parses_var_and_const() {
        let mut it = Interner::new();
        let x = parse(&mut it, "x").unwrap();
        assert_eq!(it.node(x).op, Op::Var);

        let mut it = Interner::new();
        let c = parse(&mut it, "1").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn parses_complex_literals() {
        let mut it = Interner::new();
        let c = parse(&mut it, "1+2i").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(1.0, 2.0)));

        let mut it = Interner::new();
        let c = parse(&mut it, "-3.5-0.25i").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(-3.5, -0.25)));

        let mut it = Interner::new();
        let c = parse(&mut it, "2i").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(0.0, 2.0)));
    }

    #[test]
    fn parses_named_constants() {
        let mut it = Interner::new();
        let c = parse(&mut it, "i").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(0.0, 1.0)));

        let mut it = Interner::new();
        let c = parse(&mut it, "pi").unwrap();
        assert_eq!(it.node(c).const_val, Some(Complex64::new(std::f64::consts::PI, 0.0)));
    }

    #[test]
    fn parses_nested_prim_calls_both_spellings() {
        let mut it = Interner::new();
        let a = parse(&mut it, "f(f(x, 1), 1)").unwrap();
        assert_eq!(it.node(a).op, Op::Prim);

        let mut it2 = Interner::new();
        let b = parse(&mut it2, "eml(eml(x,1),1)").unwrap();
        assert_eq!(sample_bytes(&it, a), sample_bytes(&it2, b), "f and eml must be synonyms");
    }

    #[test]
    fn rejects_malformed_input() {
        let mut it = Interner::new();
        assert!(parse(&mut it, "f(x, 1").is_err(), "missing close paren");
        assert!(parse(&mut it, "f(x 1)").is_err(), "missing comma");
        assert!(parse(&mut it, "y").is_err(), "unknown identifier");
        assert!(parse(&mut it, "f(x,1) extra").is_err(), "trailing garbage");
    }

    #[test]
    fn matches_manual_construction() {
        let mut it_manual = Interner::new();
        let x = it_manual.intern_var();
        let c = it_manual.intern_const(Complex64::new(1.0, 0.0));
        let manual_root = it_manual.intern_prim(x, c);

        let mut it_parsed = Interner::new();
        let parsed_root = parse(&mut it_parsed, "f(x, 1)").unwrap();

        assert_eq!(
            sample_bytes(&it_manual, manual_root),
            sample_bytes(&it_parsed, parsed_root)
        );
    }
}
