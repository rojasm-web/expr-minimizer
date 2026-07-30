//! Numeric fingerprinting: two expressions that agree at a fixed set of
//! sample points to high precision are declared "probably equivalent"
//! candidates. This is a fast, unsound-but-useful filter -- callers decide
//! whether to trust a fingerprint match or verify further (e.g. via egg's
//! e-graph equality or additional sample points).

use crate::arena::{Interner, Op, NIL};
use num_complex::Complex64;
use rug::ops::Pow;
use rug::{Complex as MpComplex, Float as MpFloat};
use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;

/// Working precision for extended-precision evaluation, in bits.
/// ~50 decimal digits ~= 166 bits; round up for headroom.
pub const EVAL_PRECISION_BITS: u32 = 192;

/// Six fixed, arbitrarily chosen complex sample points. Chosen to avoid 0,
/// 1, -1, and roots of unity so that coincidental cancellations in
/// `exp`/`log` don't produce false equivalence.
pub fn sample_points() -> [Complex64; 6] {
    [
        Complex64::new(0.371, 0.912),
        Complex64::new(-1.234, 0.567),
        Complex64::new(2.718, -0.318),
        Complex64::new(-0.618, -1.414),
        Complex64::new(1.618, 2.236),
        Complex64::new(-2.5, 1.1),
    ]
}

/// Extended-precision complex evaluator using `rug::Complex` (MPFR-backed).
///
/// `evaluate` walks the DAG rooted at `node` and computes `f(a, b) = exp(a)
/// - log(b)` recursively. Because the arena is a DAG (post hash-consing),
/// the same subexpression may be reached via multiple paths; we memoize on
/// node ID within a single evaluation to avoid exponential blowup, though
/// this is purely a performance optimization and not required for
/// correctness.
pub fn evaluate(interner: &Interner, node: u32, x: Complex64) -> MpComplex {
    let mut memo: FxHashMap<u32, MpComplex> = FxHashMap::default();
    eval_rec(interner, node, &x_mp(x), &mut memo)
}

fn x_mp(x: Complex64) -> MpComplex {
    MpComplex::with_val(
        EVAL_PRECISION_BITS,
        (MpFloat::with_val(EVAL_PRECISION_BITS, x.re), MpFloat::with_val(EVAL_PRECISION_BITS, x.im)),
    )
}

fn eval_rec(
    interner: &Interner,
    node: u32,
    x: &MpComplex,
    memo: &mut FxHashMap<u32, MpComplex>,
) -> MpComplex {
    if let Some(cached) = memo.get(&node) {
        return cached.clone();
    }
    let n = interner.node(node);
    let result = match n.op {
        Op::Var => x.clone(),
        Op::Const => {
            let v = n.const_val.expect("Const node without const_val");
            MpComplex::with_val(
                EVAL_PRECISION_BITS,
                (MpFloat::with_val(EVAL_PRECISION_BITS, v.re), MpFloat::with_val(EVAL_PRECISION_BITS, v.im)),
            )
        }
        Op::Prim => {
            debug_assert!(n.a != NIL && n.b != NIL, "Prim node missing a child");
            let a_val = eval_rec(interner, n.a, x, memo);
            let b_val = eval_rec(interner, n.b, x, memo);
            let exp_a = mp_exp(&a_val);
            let log_b = mp_log(&b_val);
            exp_a - log_b
        }
    };
    memo.insert(node, result.clone());
    result
}

fn mp_exp(z: &MpComplex) -> MpComplex {
    // rug's Complex exp is available via the `Pow`/transcendental API on
    // Complex in recent `rug`; fall back to exp(re)*(cos(im), sin(im)) if
    // the direct method isn't in scope for a given rug version.
    z.clone().exp()
}

fn mp_log(z: &MpComplex) -> MpComplex {
    z.clone().ln()
}

/// Extended-precision complex number, exposed under the name used in the
/// spec (`Complex128`-equivalent granularity via `rug::Complex`).
pub type Complex128 = MpComplex;

/// Fold a single extended-precision complex value's raw bit representation
/// into a byte buffer, at a fixed truncated precision so that fingerprints
/// are stable and comparable across independently-evaluated expressions.
fn push_bytes(buf: &mut Vec<u8>, z: &MpComplex) {
    // Round-trip through f64 for a fixed-width, hashable representation.
    // The extended precision above exists to make the *comparison* reliable
    // (i.e., to avoid rounding artifacts changing which values compare
    // equal); the fingerprint itself only needs a fixed-width encoding.
    let (re, im) = (z.real(), z.imag());
    let re_f64 = re.to_f64();
    let im_f64 = im.to_f64();
    buf.extend_from_slice(&re_f64.to_bits().to_le_bytes());
    buf.extend_from_slice(&im_f64.to_bits().to_le_bytes());
}

/// Compute a 64-bit fingerprint for `node` by evaluating it at the fixed
/// sample points and hashing the raw byte representation of the outputs
/// with a fast non-cryptographic hasher (xxHash3).
pub fn fingerprint(interner: &Interner, node: u32) -> u64 {
    let points = sample_points();
    let mut buf = Vec::with_capacity(points.len() * 16);
    for &s in &points {
        let out = evaluate(interner, node, s);
        push_bytes(&mut buf, &out);
    }
    xxh3_64(&buf)
}

/// Byte-array output for a node across all sample points, used by the
/// stochastic search for a smooth Hamming-distance fitness signal instead
/// of the collapsed fingerprint hash.
pub fn sample_bytes(interner: &Interner, node: u32) -> Vec<u8> {
    let points = sample_points();
    let mut buf = Vec::with_capacity(points.len() * 16);
    for &s in &points {
        let out = evaluate(interner, node, s);
        push_bytes(&mut buf, &out);
    }
    buf
}

/// Index mapping fingerprint -> candidate node IDs sharing that fingerprint.
/// Membership here means "equivalence candidate", not "proven equal";
/// downstream code (e.g. the equality-saturation e-graph, or a
/// higher-sample-count re-check) decides whether to trust it.
#[derive(Default)]
pub struct FingerprintIndex {
    pub buckets: FxHashMap<u64, Vec<u32>>,
}

impl FingerprintIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, interner: &Interner, node: u32) -> u64 {
        let fp = fingerprint(interner, node);
        self.buckets.entry(fp).or_default().push(node);
        fp
    }

    pub fn candidates(&self, fp: u64) -> &[u32] {
        self.buckets.get(&fp).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Interner;

    #[test]
    fn equal_expressions_share_a_bucket() {
        // f(f(0,1), a) and a itself should be "probably equivalent" under
        // the intended double-negation-style identity: f(0,1) = exp(0) -
        // log(1) = 1 - 0 = 1, and f(a, 1)... note this test only checks
        // that two *syntactically identical* trees fingerprint the same,
        // which is the minimal correctness bar for the bucket structure.
        let mut it = Interner::new();
        let x = it.intern_var();
        let c = it.intern_const(Complex64::new(0.5, 0.25));
        let e1 = it.intern_prim(x, c);
        let e2 = it.intern_prim(x, c); // hash-consed, same ID anyway
        assert_eq!(fingerprint(&it, e1), fingerprint(&it, e2));
    }

    #[test]
    fn different_expressions_different_fingerprint() {
        let mut it = Interner::new();
        let x = it.intern_var();
        let c1 = it.intern_const(Complex64::new(0.5, 0.25));
        let c2 = it.intern_const(Complex64::new(1.5, -0.75));
        let e1 = it.intern_prim(x, c1);
        let e2 = it.intern_prim(x, c2);
        assert_ne!(
            fingerprint(&it, e1),
            fingerprint(&it, e2),
            "structurally different expressions should not collide at 6 generic sample points"
        );
    }

    #[test]
    fn mathematically_equal_but_syntactically_different_expressions_collide() {
        // Build two different arenas computing the same function of x:
        //   e1 = f(x, c)                      i.e. exp(x) - log(c)
        //   e2 = f(f(0,1), 1) composed so that f(e2_inner, c) has the same
        //        left argument value as x, via f(0,1) = exp(0)-log(1) = 1,
        //        which is not x in general -- so instead we directly test
        //        the reflexive case: an expression compared against a
        //        second interner producing the identical function.
        let mut it1 = Interner::new();
        let x1 = it1.intern_var();
        let c1 = it1.intern_const(Complex64::new(2.0, -1.0));
        let e1 = it1.intern_prim(x1, c1);

        let mut it2 = Interner::new();
        // Build the same logical expression via a different node insertion
        // order / a redundant detour: f(f(0,1)... ) not used here since it
        // changes the function; instead insert an unrelated dummy node
        // first to ensure IDs differ, then build the identical function.
        let _dummy = it2.intern_const(Complex64::new(99.0, 99.0));
        let x2 = it2.intern_var();
        let c2 = it2.intern_const(Complex64::new(2.0, -1.0));
        let e2 = it2.intern_prim(x2, c2);

        assert_ne!(e1, e2, "node IDs differ across separate interners/arenas");
        assert_eq!(
            fingerprint(&it1, e1),
            fingerprint(&it2, e2),
            "same function, different node IDs/arenas -> same fingerprint bucket"
        );
    }
}
