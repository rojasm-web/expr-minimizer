//! Rewrite rules as data.
//!
//! IMPORTANT: these are placeholder rules only, to exercise the harness in
//! `saturate.rs`. They are NOT derived or validated algebraic identities --
//! per the spec, the real rule list will be supplied separately and should
//! be dropped in here (or loaded from an external config) without touching
//! any search code in `saturate.rs`. Do not treat the placeholders below as
//! mathematically meaningful; they exist only to prove the "rules are data"
//! wiring works end-to-end.

use egg::{rewrite, Rewrite};

use crate::saturate::L;

/// Two to three placeholder rules, matching the harness's expected shape:
/// `Vec<Rewrite<L, ()>>`. Replace/extend this with the real rule list.
pub fn placeholder_rules() -> Vec<Rewrite<L, ()>> {
    vec![
        // Placeholder: commute the two children of the primitive. This is
        // almost certainly NOT a valid identity for f(a,b) = exp(a)-log(b)
        // in general -- it's here purely to demonstrate that a two-sided
        // (bidirectional) rule round-trips through the harness.
        rewrite!("placeholder-commute"; "(f ?a ?b)" => "(f ?b ?a)"),
        // Placeholder: a trivial reflexive/no-op rule, useful as a sanity
        // check that the runner terminates cleanly even with a rule that
        // can't reduce cost.
        rewrite!("placeholder-noop"; "(f ?a ?b)" => "(f ?a ?b)"),
        // Placeholder for the "double negation"-style identity mentioned in
        // the spec (f(f(0,1), a) = a). Left commented out because it
        // depends on which ConstId indices `0` and `1` land on for a given
        // ConstTable, which isn't stable across arbitrary inputs -- this is
        // exactly the kind of rule that should come from the real,
        // supplied rule list (likely expressed as an egg `Condition` or a
        // dedicated rewrite generated per-ConstTable) rather than as a
        // hardcoded string pattern here.
        // rewrite!("double-neg"; "(f (f c0 c1) ?a)" => "?a"),
    ]
}
