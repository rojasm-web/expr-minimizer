//! Rewrite rules as data.
//!
//! ## Why this file is still short
//!
//! `f(a, b) = eml(a, b) = e^a - ln(b)` is the *only* binary primitive in
//! this language (see `arena::Op`, `saturate::L`, `fingerprint::eval_rec` --
//! all three agree there's no separate `exp`, `ln`, or `-` node). That
//! matters for which identities can be written here as sound, general
//! rewrite rules:
//!
//! - `e^ln(x) => x`, `ln(e^x) => x`, `x - 0 => x`, `e - (e - x) => x` are
//!   identities about `exp`, `ln`, and `-` as *standalone* operations.
//!   There's no tree shape in this grammar that isolates any of them from
//!   the others, so there's no pattern to write these as that would hold
//!   for arbitrary matched subexpressions (including ones containing `x`).
//!   Working the algebra through confirms this isn't just a syntax
//!   limitation: e.g. `f(?a, f(?t, 1))` evaluates to `exp(a) - ln(exp(t) -
//!   ln(1)) = exp(a) - t`, but the smallest `f`-tree that computes
//!   `exp(a) - t` for arbitrary `a`/`t` is `f(a, f(t, 1))` -- i.e. the
//!   *same* tree back again. There's no reduction available. Forcing a
//!   rule in anyway is exactly the "looks right, isn't" failure mode
//!   `placeholder-commute` used to be (see below) -- don't add these as
//!   rewrite rules. If you want free-variable-level cancellation like
//!   `e^ln(x) => x` for arbitrary `x`, the language needs `exp`/`ln`/`-` as
//!   distinct node types; that's a change to `arena::Op`/`saturate::L`,
//!   not a new entry here.
//! - `ln(1) => 0` and `ln(-1) => i*pi` aren't rewrite rules in this grammar
//!   at all -- they're facts about *evaluating* `f` on constant subtrees.
//!   That's handled by `saturate::ConstFold`, the e-class analysis wired
//!   into the runner in `main.rs`: it partially evaluates any subtree with
//!   no `Var` in it and merges e-classes that fold to the same complex
//!   value (within `saturate::CONST_FOLD_TOLERANCE`), so `ln(1) = 0` and
//!   `ln(-1) = i*pi` fall out correctly for *any* constant subtree without
//!   needing a rule for each one.
//!
//! So: this file stays a data-driven placeholder list (per the original
//! spec -- "the real rule list will be supplied separately"), plus this
//! explanation, until there's an actual *structural* identity to add (one
//! that's true for arbitrary matched subexpressions, not just constants).

use egg::{rewrite, Rewrite};

use crate::saturate::{ConstFold, L};

/// Placeholder rules, matching the harness's expected shape:
/// `Vec<Rewrite<L, ConstFold>>`. Replace/extend this with real structural
/// rules as they're identified -- see the module doc above for why the
/// exp/ln inverse identities aren't here as rewrite rules.
pub fn placeholder_rules() -> Vec<Rewrite<L, ConstFold>> {
    vec![
        // NOTE: a "placeholder-commute" rule (f a b => f b a) used to live
        // here. It was explicitly documented as almost certainly invalid
        // for the non-commutative f(a,b) = exp(a)-log(b), and it turned out
        // to actually bite: with OpCountCost giving both orderings equal
        // cost, the extractor's tie-break could return the commuted
        // (wrong) expression. Removed until a real, validated rule
        // justifies merging those e-classes. `main.rs::minimize_expr` also
        // double-checks the extractor's pick against the original's
        // fingerprint as defense in depth, in case a future rule is
        // similarly unsound.
        //
        // Placeholder: a trivial reflexive/no-op rule, useful as a sanity
        // check that the runner terminates cleanly even with a rule that
        // can't reduce cost.
        rewrite!("placeholder-noop"; "(f ?a ?b)" => "(f ?a ?b)"),
    ]
}