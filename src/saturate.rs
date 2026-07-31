//! Equality-saturation harness built on the `egg` crate.
//!
//! This module is deliberately generic over the *rule set*: rules are data
//! (`Vec<Rewrite<L, ConstFold>>`), loaded from `rules.rs`, so the search code
//! never hardcodes a specific algebraic identity. `saturate.rs` only knows
//! how to (a) translate the arena IR into egg's `RecExpr`, (b) run
//! saturation -- now paired with an e-class `Analysis` that partially
//! evaluates constant subtrees, so numerically-equal-but-syntactically-
//! unrelated constants get merged even when no rewrite rule connects them
//! -- and (c) extract a minimal-cost result by a pluggable cost function.

use std::time::Duration;

use egg::{
    define_language, Analysis, CostFunction, DidMerge, EGraph, Extractor, Id, Language, RecExpr,
    Rewrite, Runner,
};
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
#[derive(Default, Clone, Debug)]
pub struct ConstTable {
    values: Vec<Complex64>,
}

impl ConstTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exact-bit-pattern lookup/insert. Used for the initial arena -> egg
    /// translation, where two constants should only be considered "the
    /// same" if they really are bit-identical.
    pub fn intern(&mut self, v: Complex64) -> ConstId {
        if let Some(pos) = self.values.iter().position(|&existing| {
            existing.re.to_bits() == v.re.to_bits() && existing.im.to_bits() == v.im.to_bits()
        }) {
            return ConstId(pos as u32);
        }
        let id = ConstId(self.values.len() as u32);
        self.values.push(v);
        id
    }

    /// Like `intern`, but treats two values as "the same constant" once
    /// they're within `tol` of each other (Euclidean distance in the
    /// complex plane), instead of requiring an exact bit-pattern match.
    ///
    /// This is what lets `ConstFold` merge e.g. two independently-derived
    /// `i*pi` subtrees: they reach the same mathematical value via
    /// different floating-point paths, so their bit patterns won't match
    /// exactly, but `intern_approx` still hands back the *same* `ConstId` --
    /// and therefore the same interned `L::Const` e-node, and therefore
    /// the same e-class.
    pub fn intern_approx(&mut self, v: Complex64, tol: f64) -> ConstId {
        if let Some(pos) = self.values.iter().position(|&existing| (existing - v).norm() <= tol) {
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

/// Float tolerance used when deciding two folded constant values count as
/// "the same" for e-class merging. `1e-12` is comfortably above f64's ULP
/// noise floor for the kind of exp/ln chains this language produces, while
/// still tight enough that it won't accidentally conflate two genuinely
/// different constants.
pub const CONST_FOLD_TOLERANCE: f64 = 1e-12;

/// Default cap on total e-graph size passed to `Runner::with_node_limit`.
/// `ConstFold::modify` adds at most one new node per e-class whose value
/// becomes known, and `ConstTable::intern_approx` dedups re-discoveries of
/// the same folded value onto one shared `ConstId` -- so graph growth from
/// folding scales with the number of *distinct* folded values reachable,
/// not with the number of rewrite iterations. Still worth a hard ceiling
/// so a pathological interaction (or a future unsound rule) can't grow the
/// graph unboundedly.
pub const DEFAULT_NODE_LIMIT: usize = 50_000;

/// Default cap on saturation iterations, independent of the node-count
/// cap -- bounds rewrite-application work even in a run that stays small
/// but keeps discovering marginally different equivalents each pass.
pub const DEFAULT_ITER_LIMIT: usize = 60;

/// `egg::Analysis` that partially evaluates `f(a, b) = exp(a) - log(b)`
/// bottom-up: any e-class whose subtree contains no `Var` collapses to a
/// concrete `Complex64`.
///
/// Whenever an e-class's folded value becomes known, `modify` interns a
/// canonical constant node for that value -- via `ConstTable::intern_approx`,
/// so near-equal floats reached via different rewrite paths share one
/// `ConstId` -- and unions it into the class. This is what lets two
/// differently-shaped constant subtrees (e.g. one that's `f`-nested down to
/// `i*pi`, and one that's a literal `i*pi` constant) land in the same
/// e-class: there's no syntactic rewrite *rule* connecting those two node
/// shapes, and pure rewriting will never discover the connection on its
/// own within any iteration/node budget -- it's only true because of what
/// they evaluate to, which is exactly what this analysis checks instead of
/// relying on rewriting to stumble onto it.
#[derive(Debug, Clone, Default)]
pub struct ConstFold {
    pub consts: ConstTable,
}

impl ConstFold {
    pub fn new(consts: ConstTable) -> Self {
        ConstFold { consts }
    }
}

impl Analysis<L> for ConstFold {
    type Data = Option<Complex64>;

    fn make(egraph: &EGraph<L, ConstFold>, enode: &L) -> Self::Data {
        let value = |i: &Id| egraph[*i].data;
        match enode {
            L::Var => None,
            L::Const(cid) => Some(egraph.analysis.consts.get(*cid)),
            L::Prim([a, b]) => {
                let va = value(a)?;
                let vb = value(b)?;
                Some(va.exp() - vb.ln())
            }
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        match (&*a, &b) {
            (Some(_), None) => DidMerge(false, true),
            (None, Some(_)) => {
                *a = b;
                DidMerge(true, false)
            }
            (None, None) => DidMerge(false, false),
            (Some(av), Some(bv)) => {
                // Two enodes in the same e-class folding to genuinely
                // different values would mean an unsound rewrite equated
                // them -- not something this analysis can fix, but worth
                // surfacing loudly in debug builds rather than silently
                // picking one.
                debug_assert!(
                    (av - bv).norm() <= CONST_FOLD_TOLERANCE * 1e3,
                    "e-class merge of two different constant folds ({av} vs {bv}) -- \
                     this points at an unsound rewrite rule, not a bug in ConstFold"
                );
                DidMerge(false, false)
            }
        }
    }

    fn modify(egraph: &mut EGraph<L, ConstFold>, id: Id) {
        if let Some(val) = egraph[id].data {
            if val.re.is_finite() && val.im.is_finite() {
                let cid = egraph.analysis.consts.intern_approx(val, CONST_FOLD_TOLERANCE);
                let added = egraph.add(L::Const(cid));
                egraph.union(id, added);
            }
        }
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

/// Run equality saturation on `expr` using `rules` and a `ConstFold`
/// analysis seeded from `consts`, within `time_budget`, then extract the
/// minimal-cost equivalent expression under `OpCountCost`.
///
/// Returns the extracted expression *and* the (possibly-grown) constant
/// table: `ConstFold::modify` may have interned new folded constants
/// during the run, and callers need those to translate the result back to
/// the arena IR via `recexpr_to_arena`.
pub fn minimize(
    expr: RecExpr<L>,
    rules: &[Rewrite<L, ConstFold>],
    consts: ConstTable,
    time_budget: Duration,
) -> (RecExpr<L>, ConstTable) {
    let runner = Runner::<L, ConstFold>::new(ConstFold::new(consts))
        .with_expr(&expr)
        .with_time_limit(time_budget)
        .with_node_limit(DEFAULT_NODE_LIMIT)
        .with_iter_limit(DEFAULT_ITER_LIMIT)
        .run(rules);
    let extractor = Extractor::new(&runner.egraph, OpCountCost);
    let (_best_cost, best) = extractor.find_best(runner.roots[0]);
    (best, runner.egraph.analysis.consts.clone())
}

/// Same as `minimize`, but with a caller-supplied cost function, so callers
/// can swap in `ByteSizeCost` or any other `CostFunction<L>` impl.
pub fn minimize_with_cost<CF>(
    expr: RecExpr<L>,
    rules: &[Rewrite<L, ConstFold>],
    consts: ConstTable,
    time_budget: Duration,
    cost_fn: CF,
) -> (RecExpr<L>, ConstTable)
where
    CF: CostFunction<L>,
{
    let runner = Runner::<L, ConstFold>::new(ConstFold::new(consts))
        .with_expr(&expr)
        .with_time_limit(time_budget)
        .with_node_limit(DEFAULT_NODE_LIMIT)
        .with_iter_limit(DEFAULT_ITER_LIMIT)
        .run(rules);
    let extractor = Extractor::new(&runner.egraph, cost_fn);
    let (_best_cost, best) = extractor.find_best(runner.roots[0]);
    (best, runner.egraph.analysis.consts.clone())
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
        let (out, _consts) = minimize(expr, &rules, consts, Duration::from_millis(200));
        // With placeholder rules this may or may not shrink; the point of
        // this test is that the harness runs to completion without panicking.
        assert!(!out.as_ref().is_empty());
    }

    #[test]
    fn const_fold_unifies_equal_values_from_different_shapes() {
        // f(0, -1) = exp(0) - ln(-1) = 1 - i*pi. Build that as a `Prim`
        // tree, and separately build a literal `Const` equal to the same
        // value, as two independent `RecExpr`s, then add both into one
        // e-graph. Nothing in `rules.rs` relates these two node shapes
        // syntactically -- only `ConstFold::modify` can discover they're
        // equal, which is exactly the ID-15/ID-19 `i*pi` scenario this
        // analysis exists to fix.
        let mut consts = ConstTable::new();
        let zero = consts.intern(Complex64::new(0.0, 0.0));
        let neg_one = consts.intern(Complex64::new(-1.0, 0.0));
        let expected = Complex64::new(0.0, 0.0).exp() - Complex64::new(-1.0, 0.0).ln();
        let literal = consts.intern(expected);

        let mut computed_expr = RecExpr::default();
        let c_zero = computed_expr.add(L::Const(zero));
        let c_neg_one = computed_expr.add(L::Const(neg_one));
        computed_expr.add(L::Prim([c_zero, c_neg_one]));

        let mut literal_expr = RecExpr::default();
        literal_expr.add(L::Const(literal));

        let analysis = ConstFold::new(consts);
        let mut egraph: EGraph<L, ConstFold> = EGraph::new(analysis);
        let computed_id = egraph.add_expr(&computed_expr);
        let literal_id = egraph.add_expr(&literal_expr);
        egraph.rebuild();

        assert_eq!(
            egraph.find(computed_id),
            egraph.find(literal_id),
            "f(0, -1) and its literal folded value should land in the same e-class"
        );
    }
}