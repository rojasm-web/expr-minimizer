//! Generic term-rewriting / equality-saturation engine over an opaque
//! binary-op arena IR. See README / spec for the full task breakdown.
//!
//! This crate deliberately does not interpret what the composed
//! expressions "mean" -- it only knows `Op::Prim` is a binary operator
//! over two child indices, `Op::Const` is a literal leaf, and `Op::Var` is
//! the single free-variable leaf.

pub mod arena;
pub mod fingerprint;
pub mod rules;
pub mod saturate;
pub mod stochastic;

use std::time::{Duration, Instant};

use egg::StopReason;

use arena::{Arena, Interner};
use fingerprint::sample_bytes;
use saturate::{arena_to_recexpr, recexpr_to_arena, ConstTable, OpCountCost};

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
/// 2. Run equality saturation (`saturate::minimize`) with a slice of the
///    budget. If the runner reports it saturated (fully explored the
///    equivalence class under the given rules) before the time limit, its
///    extraction result is trusted directly.
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

    let runner = egg::Runner::default()
        .with_expr(&rec_expr)
        .with_time_limit(saturation_budget)
        .run(&rules);

    let saturated = matches!(runner.stop_reason, Some(StopReason::Saturated));

    let extractor = egg::Extractor::new(&runner.egraph, OpCountCost);
    let (_cost, best_rec_expr) = extractor.find_best(runner.roots[0]);
    let mut best = recexpr_to_arena(&best_rec_expr, &consts, &mut interner);

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

fn main() {
    // This crate is primarily a library (see `minimize_expr`); the binary
    // entry point exists for smoke-testing during development.
    let mut interner = Interner::new();
    let x = interner.intern_var();
    let c = interner.intern_const(num_complex::Complex64::new(1.0, 0.0));
    let root = interner.intern_prim(x, c);
    let (arena, out_root, op_count) =
        minimize_expr(&interner.arena, root, Duration::from_millis(500));
    println!(
        "minimized: {} nodes total, root = {}, op_count = {}",
        arena.len(),
        out_root,
        op_count
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

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
