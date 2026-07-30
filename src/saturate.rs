//! Equality-saturation harness built on the `egg` crate.
//!
//! This module is deliberately generic over the *rule set*: rules are data
//! (`Vec<Rewrite<L, ()>>`), loaded from `rules.rs`, so the search code never
//! hardcodes a specific algebraic identity. `saturate.rs` only knows how to
//! (a) translate the arena IR into egg's `RecExpr`, (b) run saturation, and
//! (c) extract a minimal-cost result by a pluggable cost function.

use std::time::Duration;

use egg::{define_language, CostFunction, Extractor, Id, Language, RecExpr, Rewrite, Runner};
use num_complex::Complex64;

use crate::arena::{Arena, Interner, Op as ArenaOp, NIL};

/// Opaque index into a side-table of complex constants. `egg`'s
/// `define_language!` requires leaf data to implement `Display` +
/// `FromStr` (for its s-expression parser/printer) plus the usual
/// `Hash + Eq + Ord + Clone + Debug`; `Complex64` doesn't implement `Ord`,
/// so constants are represented indirectly by an index into `ConstTable`
/// rather than embedding the float pair directly in the language.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ConstId(pub u32);

impl std::fmt::Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "c{}", self.0)
    }
}

impl std::str::FromStr for ConstId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.strip_prefix('c')
            .and_then(|rest| rest.parse::<u32>().ok())
            .map(ConstId)
            .ok_or_else(|| format!("invalid ConstId literal: {s}"))
    }
}

/// Side table mapping `ConstId -> Complex64`, shared between the arena and
/// the egg `Language` translation so constant identity survives round-trips.
#[derive(Default, Clone)]
pub struct ConstTable {
    values: Vec<Complex64>,
}

impl ConstTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, v: Complex64) -> ConstId {
        // Exact-bit-pattern lookup, mirroring Interner::intern_const.
        if let Some(pos) = self.values.iter().position(|&existing| {
            existing.re.to_bits() == v.re.to_bits() && existing.im.to_bits() == v.im.to_bits()
        }) {
            return ConstId(pos as u32);
        }
        let id = ConstId(self.values.len() as u32);
        self.values.push(v);
        id
    }

    pub fn get(&self, id: ConstId) -> Complex64 {
        self.values[id.0 as usize]
    }
}

define_language! {
    /// Mirrors `arena::Op`: a single binary primitive, a constant leaf
    /// (indirected through `ConstTable`), and the free variable leaf.
    pub enum L {
        "f" = Prim([Id; 2]),
        Const(ConstId),
        "x" = Var,
    }
}

/// Translate an arena-rooted expression into an egg `RecExpr<L>`.
pub fn arena_to_recexpr(interner: &Interner, root: u32, consts: &mut ConstTable) -> RecExpr<L> {
    let mut expr = RecExpr::default();
    let mut memo: rustc_hash::FxHashMap<u32, Id> = rustc_hash::FxHashMap::default();
    to_recexpr_rec(interner, root, consts, &mut expr, &mut memo);
    expr
}

fn to_recexpr_rec(
    interner: &Interner,
    node: u32,
    consts: &mut ConstTable,
    expr: &mut RecExpr<L>,
    memo: &mut rustc_hash::FxHashMap<u32, Id>,
) -> Id {
    if let Some(&id) = memo.get(&node) {
        return id;
    }
    let n = interner.node(node);
    let id = match n.op {
        ArenaOp::Var => expr.add(L::Var),
        ArenaOp::Const => {
            let v = n.const_val.expect("Const node without const_val");
            let cid = consts.intern(v);
            expr.add(L::Const(cid))
        }
        ArenaOp::Prim => {
            debug_assert!(n.a != NIL && n.b != NIL);
            let a = to_recexpr_rec(interner, n.a, consts, expr, memo);
            let b = to_recexpr_rec(interner, n.b, consts, expr, memo);
            expr.add(L::Prim([a, b]))
        }
    };
    memo.insert(node, id);
    id
}

/// Translate an egg `RecExpr<L>` (typically the extraction result) back
/// into the arena IR, re-interning through `Interner` so the result stays
/// hash-consed.
pub fn recexpr_to_arena(expr: &RecExpr<L>, consts: &ConstTable, interner: &mut Interner) -> u32 {
    let nodes = expr.as_ref();
    let mut built: Vec<u32> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let id = match node {
            L::Var => interner.intern_var(),
            L::Const(cid) => interner.intern_const(consts.get(*cid)),
            L::Prim([a, b]) => {
                let a_id = built[usize::from(*a)];
                let b_id = built[usize::from(*b)];
                interner.intern_prim(a_id, b_id)
            }
        };
        built.push(id);
    }
    *built.last().expect("empty RecExpr")
}

/// Cost function: total primitive-op count (1 per leaf, `1 + cost(a) +
/// cost(b)` per `Prim` node). Exposed as a pluggable `CostFunction` impl so
/// it can be swapped later (e.g. for serialized byte size).
pub struct OpCountCost;

impl CostFunction<L> for OpCountCost {
    type Cost = u64;

    fn cost<C>(&mut self, enode: &L, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let child_cost: u64 = enode.children().iter().map(|&id| costs(id)).sum();
        1 + child_cost
    }
}

/// Alternative pluggable cost: serialized byte size instead of op count.
/// `Prim` nodes cost 1 (opcode tag) + 2 * size_of(Id) for the two child
/// references; leaves cost a fixed encoding size.
pub struct ByteSizeCost;

impl CostFunction<L> for ByteSizeCost {
    type Cost = u64;

    fn cost<C>(&mut self, enode: &L, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        const ID_BYTES: u64 = 4;
        const TAG_BYTES: u64 = 1;
        match enode {
            L::Var => TAG_BYTES,
            L::Const(_) => TAG_BYTES + 16, // tag + re/im f64 pair
            L::Prim([a, b]) => TAG_BYTES + 2 * ID_BYTES + costs(*a) + costs(*b),
        }
    }
}

/// Run equality saturation on `expr` using `rules`, within `time_budget`,
/// then extract the minimal-cost equivalent expression under `OpCountCost`.
pub fn minimize(expr: RecExpr<L>, rules: &[Rewrite<L, ()>], time_budget: Duration) -> RecExpr<L> {
    let runner = Runner::default()
        .with_expr(&expr)
        .with_time_limit(time_budget)
        .run(rules);
    let extractor = Extractor::new(&runner.egraph, OpCountCost);
    let (_best_cost, best) = extractor.find_best(runner.roots[0]);
    best
}

/// Same as `minimize`, but with a caller-supplied cost function, so callers
/// can swap in `ByteSizeCost` or any other `CostFunction<L>` impl.
pub fn minimize_with_cost<CF>(
    expr: RecExpr<L>,
    rules: &[Rewrite<L, ()>],
    time_budget: Duration,
    cost_fn: CF,
) -> RecExpr<L>
where
    CF: CostFunction<L>,
{
    let runner = Runner::default()
        .with_expr(&expr)
        .with_time_limit(time_budget)
        .run(rules);
    let extractor = Extractor::new(&runner.egraph, cost_fn);
    let (_best_cost, best) = extractor.find_best(runner.roots[0]);
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::placeholder_rules;

    #[test]
    fn round_trip_arena_recexpr_arena() {
        let mut interner = Interner::new();
        let x = interner.intern_var();
        let c = interner.intern_const(Complex64::new(1.0, 0.0));
        let root = interner.intern_prim(x, c);

        let mut consts = ConstTable::new();
        let expr = arena_to_recexpr(&interner, root, &mut consts);

        let mut interner2 = Interner::new();
        let round_tripped = recexpr_to_arena(&expr, &consts, &mut interner2);
        let node = interner2.node(round_tripped);
        assert_eq!(node.op, ArenaOp::Prim);
    }

    #[test]
    fn saturation_harness_runs_with_placeholder_rules() {
        let mut interner = Interner::new();
        let x = interner.intern_var();
        let one = interner.intern_const(Complex64::new(1.0, 0.0));
        let root = interner.intern_prim(x, one);

        let mut consts = ConstTable::new();
        let expr = arena_to_recexpr(&interner, root, &mut consts);
        let rules = placeholder_rules();
        let out = minimize(expr, &rules, Duration::from_millis(200));
        // With placeholder rules this may or may not shrink; the point of
        // this test is that the harness runs to completion without panicking.
        assert!(!out.as_ref().is_empty());
    }
}
