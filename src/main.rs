//! Generic term-rewriting / equality-saturation engine over an opaque
//! binary-op arena IR. See README / spec for the full task breakdown.
//!
//! This crate deliberately does not interpret what the composed
//! expressions "mean" -- it only knows `Op::Prim` is a binary operator
//! over two child indices, `Op::Const` is a literal leaf, and `Op::Var` is
//! the single free-variable leaf.
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

pub mod arena;
pub mod equiv;
pub mod fingerprint;
pub mod parser;
pub mod rules;
pub mod saturate;
pub mod stochastic;
pub mod catalog;
pub mod config;

use egg::{Runner, StopReason};

use arena::{Arena, Interner};
use fingerprint::sample_bytes;
use saturate::{
    arena_to_recexpr, recexpr_to_arena, ConstFold, ConstTable, OpCountCost, DEFAULT_ITER_LIMIT,
    DEFAULT_NODE_LIMIT, L,
};

/// Copy every node of `src` into a fresh, hash-consed `Interner`, returning
/// the interner plus an `old_id -> new_id` mapping. Because `src` is
/// assumed to already be a valid arena (children indices are always less
/// than their parent's index, as produced by `Interner::intern_*`), a
/// single forward pass suffices.
fn rehydrate(src: &Arena) -> (Interner, Vec<u32>) {
    let mut interner = Interner::new();
    let mut map = vec![arena::NIL; src.len()];
    for (old_id, node) in src.nodes.iter().enumerate() {
        let new_id = match node.op {
            arena::Op::Var => interner.intern_var(),
            arena::Op::Const => {
                interner.intern_const(node.const_val.expect("Const node without const_val"))
            }
            arena::Op::Prim => {
                let a = map[node.a as usize];
                let b = map[node.b as usize];
                debug_assert!(a != arena::NIL && b != arena::NIL, "child must precede parent");
                interner.intern_prim(a, b)
            }
        };
        map[old_id] = new_id;
    }
    (interner, map)
}

/// Minimize the expression rooted at `root` within `expr`, spending up to
/// `budget` total. Strategy:
///
/// 1. Rehydrate the input arena into a hash-consed `Interner`.
/// 2. Run equality saturation, paired with the `ConstFold` e-class
///    analysis (see `saturate.rs`), with a slice of the budget. If the
///    runner reports it saturated (fully explored the equivalence class
///    under the given rules, not just hit a time/node/iteration limit)
///    before any limit, its extraction result is trusted directly.
/// 3. Otherwise -- saturation didn't converge -- spend the remaining
///    budget on the stochastic fallback search (`stochastic::search`),
///    seeded from whatever equality saturation already found, targeting
///    the *original* expression's sample-point fingerprint so functional
///    equivalence is preserved throughout.
/// 4. Return a fresh, minimal `Arena` containing just the winning
///    expression, its root ID within that arena, and its op count.
pub fn minimize_expr(expr: &Arena, root: u32, budget: Duration) -> (Arena, u32, u64) {
    let start = Instant::now();
    let (mut interner, id_map) = rehydrate(expr);
    let root = id_map[root as usize];

    let target_bytes = sample_bytes(&interner, root);

    // Give equality saturation a majority slice of the budget; the
    // stochastic fallback gets whatever remains.
    let saturation_budget = budget.mul_f64(0.7);

    let mut consts = ConstTable::new();
    let rec_expr = arena_to_recexpr(&interner, root, &mut consts);
    let rules = rules::placeholder_rules();

    // Node/iteration limits sit alongside the time limit so that
    // ConstFold's constant-merging can't run away with the graph size --
    // see saturate::DEFAULT_NODE_LIMIT / DEFAULT_ITER_LIMIT for why those
    // particular defaults are safe headroom rather than arbitrary caps.
    // Hitting any of the three limits (time, nodes, iterations) without
    // reaching `StopReason::Saturated` is treated identically below: the
    // extraction still happens, but it isn't trusted without the
    // fingerprint check, and the stochastic fallback still gets a turn.
    let runner = Runner::<L, ConstFold>::new(ConstFold::new(consts))
        .with_expr(&rec_expr)
        .with_time_limit(saturation_budget)
        .with_node_limit(DEFAULT_NODE_LIMIT)
        .with_iter_limit(DEFAULT_ITER_LIMIT)
        .run(&rules);

    let saturated = matches!(runner.stop_reason, Some(StopReason::Saturated));

    // `ConstFold::modify` may have interned new folded constants during
    // the run (e.g. a freshly-discovered `i*pi`), so pull the updated
    // table back out rather than reusing the pre-run `consts`.
    let consts = runner.egraph.analysis.consts.clone();

    let extractor = egg::Extractor::new(&runner.egraph, OpCountCost);
    let (_cost, best_rec_expr) = extractor.find_best(runner.roots[0]);
    let mut best = recexpr_to_arena(&best_rec_expr, &consts, &mut interner);

    // Never trust a rewrite rule set's output blindly: verify the
    // extractor's pick actually still computes the original function
    // before treating it as a candidate. Cost-based tie-breaking can't
    // distinguish "valid rewrite, different price" from "outright wrong,
    // same price" -- an unsound rule (or a bug in a future one) would
    // otherwise slip through silently. If it doesn't match, fall back to
    // the verified-correct pre-saturation root and let the stochastic
    // search (which does its own exact-match verification) take over.
    if sample_bytes(&interner, best) != target_bytes {
        best = root;
    }

    let elapsed = start.elapsed();
    if !saturated && elapsed < budget {
        let remaining = budget - elapsed;
        let config = stochastic::SearchConfig::default();
        let result = stochastic::search(&mut interner, best, &target_bytes, &config, remaining);
        if result.exact_match_found && result.best_fitness
            > -( stochastic::op_count(&interner, best) as f64 * config.lambda)
        {
            best = result.best;
        }
    }

    // Final cleanup pass: independently of whatever rules.rs supplied,
    // scan the winning expression bottom-up and collapse any subtrees that
    // are numerically equivalent (agree at all sample points) down to
    // whichever is cheapest, regardless of whether an explicit algebraic
    // rule justifies the identity. This catches equivalences saturation's
    // rule set doesn't know about, and handles equivalence classes with
    // more than two members via `equiv::collapse_equivalent_subtrees`'s
    // fixed-point iteration.
    best = equiv::collapse_equivalent_subtrees(&mut interner, best, 8);

    // Materialize a fresh, minimal arena containing only the winning
    // expression's reachable nodes, in a stable topological (child-before-
    // parent) order, matching the invariant `rehydrate` relies on.
    let (out_arena, out_root) = extract_minimal_arena(&interner, best);
    let op_count = stochastic::op_count(&interner, best);
    (out_arena, out_root, op_count)
}

/// Walk `root`'s reachable subtree in post-order and rebuild it into a
/// fresh, compact `Arena` (dropping unrelated nodes left over from the
/// search population).
fn extract_minimal_arena(interner: &Interner, root: u32) -> (Arena, u32) {
    let mut out = Interner::new();
    let mut memo: rustc_hash::FxHashMap<u32, u32> = rustc_hash::FxHashMap::default();
    let new_root = extract_rec(interner, root, &mut out, &mut memo);
    (out.arena, new_root)
}

fn extract_rec(
    interner: &Interner,
    node: u32,
    out: &mut Interner,
    memo: &mut rustc_hash::FxHashMap<u32, u32>,
) -> u32 {
    if let Some(&id) = memo.get(&node) {
        return id;
    }
    let n = interner.node(node);
    let id = match n.op {
        arena::Op::Var => out.intern_var(),
        arena::Op::Const => out.intern_const(n.const_val.expect("Const node without const_val")),
        arena::Op::Prim => {
            let a = extract_rec(interner, n.a, out, memo);
            let b = extract_rec(interner, n.b, out, memo);
            out.intern_prim(a, b)
        }
    };
    memo.insert(node, id);
    id
}

fn print_tree(interner: &Interner, node_id: u32, depth: usize) {
    let indent = "  ".repeat(depth);
    let node = interner.node(node_id);

    match node.op {
        arena::Op::Var => {
            println!("{}x (id: {})", indent, node_id);
        }
        arena::Op::Const => {
            let val = node.const_val.expect("Const missing value");
            println!("{}Const({}) (id: {})", indent, val, node_id);
        }
        arena::Op::Prim => {
            println!("{}eml (id: {})", indent, node_id);
            print_tree(interner, node.a, depth + 1);
            print_tree(interner, node.b, depth + 1);
        }
    }
}

fn to_expr_string(interner: &Interner, node_id: u32) -> String {
    let node = interner.node(node_id);
    match node.op {
        arena::Op::Var => "x".to_string(),
        arena::Op::Const => {
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
        arena::Op::Prim => {
            let left = to_expr_string(interner, node.a);
            let right = to_expr_string(interner, node.b);
            format!("eml({},{})", left, right)
        }
    }
}

fn main() {
    let mut interner = Interner::new();

    // 1. Parse input expression
    let input_str = "eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(eml(eml(eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1)),1)))),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1))),1))),1)),1)),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),x),1)),1),eml(eml(eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(eml(eml(eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1)),1)))),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1))),1))),1)),1)),1)),1),1),1))),1))),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),x),1)),1),1),1))),1))),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1)),1))),1))),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(eml(eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1)),1)))),1)),eml(eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(1,eml(1,eml(eml(1,1),1)))),eml(1,1))),1))),1)),1)),1)),1),1)),1)),1)),1)"
        .to_string();
    let root = parser::parse(&mut interner, &input_str).expect("valid expression");

    println!("=== INPUT EML TREE ===");
    print_tree(&interner, root, 0);
    println!("\nTotal nodes: {}", interner.arena.len());
    println!("Op count: {}\n", stochastic::op_count(&interner, root));

    // 2. Minimize expression
    let (min_arena, min_root, min_op_count) =
        minimize_expr(&interner.arena, root, Duration::from_millis(500));

    // 3. Rehydrate output arena
    let (min_interner, _) = rehydrate(&min_arena);

    println!("=== MINIMIZED EML TREE ===");
    print_tree(&min_interner, min_root, 0);
    println!("\nMinimized total nodes: {}", min_arena.len());
    println!("Minimized op count: {}", min_op_count);

    // 4. Save single-line strings to expressions.txt
    let og_single_line = to_expr_string(&interner, root);
    let min_single_line = to_expr_string(&min_interner, min_root);
    catalog::export_constant_catalog(&min_interner, min_root, "constants_catalog.txt");

    let mut file = File::create("expressions.txt").expect("failed to create expressions.txt");
    writeln!(file, "ORIGINAL:\n{}", og_single_line).unwrap();
    writeln!(file, "\nMINIMIZED:\n{}", min_single_line).unwrap();

    println!("\nSaved single-line representations to `expressions.txt`.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn minimize_expr_accepts_parsed_text_input() {
        let mut interner = Interner::new();
        let root = parser::parse(&mut interner, "f(f(x, 1), 1+0i)").expect("valid expression");
        let before_bytes = sample_bytes(&interner, root);

        let (out_arena, out_root, _op_count) =
            minimize_expr(&interner.arena, root, Duration::from_millis(300));

        let (out_interner, _) = rehydrate(&out_arena);
        let after_bytes = sample_bytes(&out_interner, out_root);
        assert_eq!(
            before_bytes, after_bytes,
            "minimization of parsed input must not change the expression's function"
        );
    }

    #[test]
    fn minimize_expr_preserves_functional_behavior() {
        let mut interner = Interner::new();
        let x = interner.intern_var();
        let c = interner.intern_const(Complex64::new(1.0, 0.0));
        let root = interner.intern_prim(x, c);
        let before_bytes = sample_bytes(&interner, root);

        let (out_arena, out_root, _op_count) =
            minimize_expr(&interner.arena, root, Duration::from_millis(300));

        let (out_interner, _) = rehydrate(&out_arena);
        let after_bytes = sample_bytes(&out_interner, out_root);
        assert_eq!(
            before_bytes, after_bytes,
            "minimization must not change the expression's function"
        );
    }
}
