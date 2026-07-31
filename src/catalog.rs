//! Symbolic Catalog Engine for identifying and interpreting constant subtrees.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::arena::{Interner, Op};

/// Symbolic AST representation used for exact algebraic evaluation and formatting.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpr {
    Num(i64),
    Float(f64),
    Complex(Complex64),
    E,
    Pi,
    I,
    Inf(bool),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Pow(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Ln(Box<SymbolicExpr>),
    Exp(Box<SymbolicExpr>),
    Neg(Box<SymbolicExpr>),
}

impl SymbolicExpr {
    /// Formats the symbolic expression cleanly with strict order-of-operations parentheses.
    pub fn format_clean(&self) -> String {
        match self {
            SymbolicExpr::Num(n) => format!("{}", n),
            SymbolicExpr::Float(f) => format!("{:.4}", f),
            SymbolicExpr::Complex(c) => {
                if c.re == 0.0 {
                    format!("{:.4}i", c.im)
                } else if c.im == 0.0 {
                    format!("{:.4}", c.re)
                } else {
                    format!("{:.4} + {:.4}i", c.re, c.im)
                }
            }
            SymbolicExpr::E => "e".to_string(),
            SymbolicExpr::Pi => "pi".to_string(),
            SymbolicExpr::I => "i".to_string(),
            SymbolicExpr::Inf(pos) => if *pos { "+inf".to_string() } else { "-inf".to_string() },
            SymbolicExpr::Neg(inner) => {
                let s = inner.format_clean();
                if matches!(**inner, SymbolicExpr::Add(_, _) | SymbolicExpr::Sub(_, _)) {
                    format!("-({})", s)
                } else {
                    format!("-{}", s)
                }
            }
            SymbolicExpr::Add(a, b) => format!("{} + {}", a.format_clean(), b.format_clean()),
            SymbolicExpr::Sub(a, b) => format!("{} - {}", a.format_clean(), b.format_parenthesized()),

            SymbolicExpr::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                (SymbolicExpr::Num(1), SymbolicExpr::I) | (SymbolicExpr::I, SymbolicExpr::Num(1)) => "i".to_string(),
                (SymbolicExpr::Num(-1), SymbolicExpr::I) | (SymbolicExpr::I, SymbolicExpr::Num(-1)) => "-i".to_string(),
                (SymbolicExpr::Num(n), SymbolicExpr::I) | (SymbolicExpr::I, SymbolicExpr::Num(n)) => format!("{}i", n),
                _ => format!("{} * {}", a.format_parenthesized(), b.format_parenthesized()),
            },

            SymbolicExpr::Div(a, b) => format!("{}/{}", a.format_parenthesized(), b.format_parenthesized()),
            SymbolicExpr::Pow(a, b) => format!("{}^{}", a.format_parenthesized(), b.format_parenthesized()),
            SymbolicExpr::Ln(a) => format!("ln({})", a.format_clean()),
            SymbolicExpr::Exp(a) => format!("e^{}", a.format_parenthesized()),
        }
    }

    fn format_parenthesized(&self) -> String {
        match self {
            SymbolicExpr::Add(_, _) | SymbolicExpr::Sub(_, _) | SymbolicExpr::Mul(_, _) | SymbolicExpr::Div(_, _) => {
                format!("({})", self.format_clean())
            }
            _ => self.format_clean(),
        }
    }
}

/// Evaluates an interner EML node directly to a complex float: eml(a, b) = exp(a) - ln(b).
pub fn eval_numerical_direct(
    interner: &Interner,
    node_id: u32,
    memo: &mut FxHashMap<u32, Complex64>,
) -> Complex64 {
    if let Some(&val) = memo.get(&node_id) {
        return val;
    }

    let node = interner.node(node_id);
    let val = match node.op {
        Op::Var => panic!("Cannot evaluate variable node numerically as constant"),
        Op::Const => node.const_val.expect("Const node missing val"),
        Op::Prim => {
            let a_val = eval_numerical_direct(interner, node.a, memo);
            let b_val = eval_numerical_direct(interner, node.b, memo);
            a_val.exp() - b_val.ln()
        }
    };

    memo.insert(node_id, val);
    val
}

pub fn interpret_symbolic(interner: &Interner, node_id: u32, memo: &mut FxHashMap<u32, SymbolicExpr>) -> SymbolicExpr {
    if let Some(expr) = memo.get(&node_id) {
        return expr.clone();
    }

    let node = interner.node(node_id);
    let raw_expr = match node.op {
        Op::Var => panic!("Cannot interpret variable node symbolically as constant"),
        Op::Const => {
            let val = node.const_val.expect("Const node missing val");
            if val.im == 0.0 {
                if val.re.fract() == 0.0 {
                    SymbolicExpr::Num(val.re as i64)
                } else {
                    SymbolicExpr::Float(val.re)
                }
            } else {
                SymbolicExpr::Complex(val)
            }
        }
        Op::Prim => {
            let left_sym = interpret_symbolic(interner, node.a, memo);
            let right_sym = interpret_symbolic(interner, node.b, memo);

            SymbolicExpr::Sub(
                Box::new(SymbolicExpr::Exp(Box::new(left_sym))),
                Box::new(SymbolicExpr::Ln(Box::new(right_sym))),
            )
        }
    };

    let simplified = reduce_fixed_point(raw_expr);
    memo.insert(node_id, simplified.clone());
    simplified
}

fn reduce_fixed_point(mut expr: SymbolicExpr) -> SymbolicExpr {
    let mut max_passes = 20;
    while max_passes > 0 {
        let next = simplify_pass(expr.clone());
        if next == expr {
            break;
        }
        expr = next;
        max_passes -= 1;
    }
    expr
}

fn simplify_pass(expr: SymbolicExpr) -> SymbolicExpr {
    match expr {
        SymbolicExpr::Ln(inner) => {
            let inner_reduced = simplify_pass(*inner);
            match inner_reduced {
                SymbolicExpr::Num(1) => SymbolicExpr::Num(0),
                SymbolicExpr::Num(-1) => SymbolicExpr::Mul(Box::new(SymbolicExpr::I), Box::new(SymbolicExpr::Pi)),
                SymbolicExpr::E => SymbolicExpr::Num(1),
                SymbolicExpr::Exp(x) => *x,
                SymbolicExpr::Pow(b, x) if *b == SymbolicExpr::E => *x,
                SymbolicExpr::Num(0) => SymbolicExpr::Inf(false),
                SymbolicExpr::Inf(true) => SymbolicExpr::Inf(true),
                SymbolicExpr::I => SymbolicExpr::Div(Box::new(SymbolicExpr::Mul(Box::new(SymbolicExpr::I), Box::new(SymbolicExpr::Pi))), Box::new(SymbolicExpr::Num(2))),
                SymbolicExpr::Neg(inner) if matches!(inner.as_ref(), SymbolicExpr::I) => SymbolicExpr::Neg(Box::new(SymbolicExpr::Div(Box::new(SymbolicExpr::Mul(Box::new(SymbolicExpr::I), Box::new(SymbolicExpr::Pi))), Box::new(SymbolicExpr::Num(2))))),
                SymbolicExpr::Mul(a, b) if matches!(a.as_ref(), SymbolicExpr::Exp(_)) || matches!(b.as_ref(), SymbolicExpr::Exp(_)) => {
                    let ln_a = simplify_pass(SymbolicExpr::Ln(a));
                    let ln_b = simplify_pass(SymbolicExpr::Ln(b));
                    simplify_pass(SymbolicExpr::Add(Box::new(ln_a), Box::new(ln_b)))
                }
                other => SymbolicExpr::Ln(Box::new(other)),
            }
        }

        SymbolicExpr::Exp(inner) => {
            let inner_reduced = simplify_pass(*inner);
            match inner_reduced {
                SymbolicExpr::Num(0) => SymbolicExpr::Num(1),
                SymbolicExpr::Num(1) => SymbolicExpr::E,
                SymbolicExpr::Ln(x) => *x,
                SymbolicExpr::Inf(true) => SymbolicExpr::Inf(true),
                SymbolicExpr::Inf(false) => SymbolicExpr::Num(0),
                SymbolicExpr::Mul(a, b) if matches!((a.as_ref(), b.as_ref()), (SymbolicExpr::I, SymbolicExpr::Pi) | (SymbolicExpr::Pi, SymbolicExpr::I)) => SymbolicExpr::Num(-1),
                SymbolicExpr::Div(a, b) if matches!((b.as_ref()), (SymbolicExpr::Num(2))) && matches!(a.as_ref(), SymbolicExpr::Mul(x, y) if matches!((x.as_ref(), y.as_ref()), (SymbolicExpr::I, SymbolicExpr::Pi) | (SymbolicExpr::Pi, SymbolicExpr::I))) => SymbolicExpr::I,
                SymbolicExpr::Add(a, b) => simplify_pass(SymbolicExpr::Mul(Box::new(SymbolicExpr::Exp(a)), Box::new(SymbolicExpr::Exp(b)))),
                other => SymbolicExpr::Exp(Box::new(other)),
            }
        }

        SymbolicExpr::Sub(a, b) => {
            let a_red = simplify_pass(*a);
            let b_red = simplify_pass(*b);

            if a_red == b_red {
                return SymbolicExpr::Num(0);
            }

            match (a_red, b_red) {
                (x, SymbolicExpr::Num(0)) => x,
                (SymbolicExpr::Num(0), x) => match x {
                    SymbolicExpr::Inf(true) => SymbolicExpr::Inf(false),
                    SymbolicExpr::Inf(false) => SymbolicExpr::Inf(true),
                    SymbolicExpr::Neg(inner) => *inner,
                    _ => SymbolicExpr::Neg(Box::new(x)),
                },

                (SymbolicExpr::Num(na), SymbolicExpr::Num(nb)) => SymbolicExpr::Num(na - nb),

                (SymbolicExpr::Ln(x), SymbolicExpr::Ln(y)) => {
                    SymbolicExpr::Ln(Box::new(SymbolicExpr::Div(x, y)))
                }

                (a, SymbolicExpr::Neg(b)) => {
                    simplify_pass(SymbolicExpr::Add(Box::new(a), b))
                }

                (ref left, SymbolicExpr::Sub(ref sub_a, ref sub_b)) if left == sub_a.as_ref() => {
                    *sub_b.clone()
                }

                (_, SymbolicExpr::Inf(true)) => SymbolicExpr::Inf(false),
                (_, SymbolicExpr::Inf(false)) => SymbolicExpr::Inf(true),
                (SymbolicExpr::Inf(true), _) => SymbolicExpr::Inf(true),
                (SymbolicExpr::Inf(false), _) => SymbolicExpr::Inf(false),

                (a, SymbolicExpr::Add(b, c)) if a == *b => SymbolicExpr::Neg(c),
                (a, SymbolicExpr::Add(b, c)) if a == *c => SymbolicExpr::Neg(b),

                (l, r) => SymbolicExpr::Sub(Box::new(l), Box::new(r)),
            }
        }

        SymbolicExpr::Mul(a, b) => {
            let a_red = simplify_pass(*a);
            let b_red = simplify_pass(*b);
            match (a_red, b_red) {
                (SymbolicExpr::Num(0), _) | (_, SymbolicExpr::Num(0)) => SymbolicExpr::Num(0),
                (SymbolicExpr::Num(1), x) | (x, SymbolicExpr::Num(1)) => x,
                (SymbolicExpr::Num(na), SymbolicExpr::Num(nb)) => SymbolicExpr::Num(na * nb),
                (l, r) => SymbolicExpr::Mul(Box::new(l), Box::new(r)),
            }
        }

        SymbolicExpr::Div(a, b) => {
            let a_red = simplify_pass(*a);
            let b_red = simplify_pass(*b);
            if a_red == b_red {
                return SymbolicExpr::Num(1);
            }
            match (a_red, b_red) {
                (x, SymbolicExpr::Num(1)) => x,
                (SymbolicExpr::Num(0), _) => SymbolicExpr::Num(0),
                (SymbolicExpr::Num(na), SymbolicExpr::Num(nb)) if nb != 0 && na % nb == 0 => {
                    SymbolicExpr::Num(na / nb)
                }
                (l, r) => SymbolicExpr::Div(Box::new(l), Box::new(r)),
            }
        }

        SymbolicExpr::Neg(inner) => {
            let inner_red = simplify_pass(*inner);
            match inner_red {
                SymbolicExpr::Neg(x) => *x,
                SymbolicExpr::Num(n) => SymbolicExpr::Num(-n),
                x => SymbolicExpr::Neg(Box::new(x)),
            }
        }

        SymbolicExpr::Add(a, b) => {
            let a_red = simplify_pass(*a);
            let b_red = simplify_pass(*b);
            match (a_red, b_red) {
                (SymbolicExpr::Num(0), x) | (x, SymbolicExpr::Num(0)) => x,
                (SymbolicExpr::Num(na), SymbolicExpr::Num(nb)) => SymbolicExpr::Num(na + nb),

                (SymbolicExpr::Neg(x), y) => {
                    simplify_pass(SymbolicExpr::Sub(Box::new(y), x))
                }
                (x, SymbolicExpr::Neg(y)) => {
                    simplify_pass(SymbolicExpr::Sub(Box::new(x), y))
                }

                (l, r) => SymbolicExpr::Add(Box::new(l), Box::new(r)),
            }
        }

        other => other,
    }
}

pub fn is_constant_node(interner: &Interner, node_id: u32, memo: &mut FxHashMap<u32, bool>) -> bool {
    if let Some(&res) = memo.get(&node_id) {
        return res;
    }
    let node = interner.node(node_id);
    let res = match node.op {
        Op::Var => false,
        Op::Const => true,
        Op::Prim => {
            let left_const = is_constant_node(interner, node.a, memo);
            let right_const = is_constant_node(interner, node.b, memo);
            left_const && right_const
        }
    };
    memo.insert(node_id, res);
    res
}

fn to_eml_string(interner: &Interner, node_id: u32) -> String {
    let node = interner.node(node_id);
    match node.op {
        Op::Var => "x".to_string(),
        Op::Const => {
            let val = node.const_val.expect("Const missing value");
            if val.im == 0.0 {
                if val.re.fract() == 0.0 {
                    format!("{}", val.re as i64)
                } else {
                    format!("{}", val.re)
                }
            } else {
                format!("{}+{}i", val.re, val.im)
            }
        }
        Op::Prim => {
            let left = to_eml_string(interner, node.a);
            let right = to_eml_string(interner, node.b);
            format!("eml({},{})", left, right)
        }
    }
}

pub fn export_constant_catalog(interner: &Interner, root_id: u32, filename: &str) {
    let mut is_const_memo = FxHashMap::default();
    let mut sym_memo = FxHashMap::default();
    let mut num_memo = FxHashMap::default();
    let mut visited = HashSet::new();
    let mut catalog = Vec::new();

    let mut stack = vec![root_id];
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }

        if is_constant_node(interner, node_id, &mut is_const_memo) {
            let sym_val = interpret_symbolic(interner, node_id, &mut sym_memo);
            let num_val = eval_numerical_direct(interner, node_id, &mut num_memo);
            let eml_repr = to_eml_string(interner, node_id);

            catalog.push((node_id, sym_val, num_val, eml_repr));
        }

        let node = interner.node(node_id);
        if node.op == Op::Prim {
            stack.push(node.a);
            stack.push(node.b);
        }
    }

    catalog.sort_by_key(|(id, _, _, _)| *id);

    let mut file = File::create(filename).expect("Failed to create catalog file");
    writeln!(file, "=========================================================").unwrap();
    writeln!(file, "          SYMBOLIC CONSTANT SUBTREES CATALOG             ").unwrap();
    writeln!(file, "=========================================================\n").unwrap();

    for (id, sym_val, num_val, eml_repr) in catalog {
        writeln!(file, "id {}:", id).unwrap();
        writeln!(file, "  Symbolic Value  : {}", sym_val.format_clean()).unwrap();
        
        if num_val.im.abs() < 1e-12 {
            writeln!(file, "  Numerical Value : {:.6}", num_val.re).unwrap();
        } else if num_val.re.abs() < 1e-12 {
            writeln!(file, "  Numerical Value : {:.6}i", num_val.im).unwrap();
        } else {
            writeln!(file, "  Numerical Value : {:.6} + {:.6}i", num_val.re, num_val.im).unwrap();
        }

        writeln!(file, "  EML Subtree     : {}\n", eml_repr).unwrap();
    }

    println!("Symbolic constant catalog saved to `{}`!", filename);
}