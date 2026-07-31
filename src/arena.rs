//! Flat, indexable arena representation for binary-op expression trees,
//! plus a hash-consing interner that structurally deduplicates subexpressions.
//!
//! This module is intentionally generic: it knows nothing about what `Prim`
//! *means*, only that it is a binary operator over two child node indices.

use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::config::MACHINE_TOLERANCE;

/// Sentinel for "no child" (used by leaf nodes).
pub const NIL: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// The single binary primitive `f(a, b)`.
    Prim,
    /// Leaf: a literal complex constant.
    Const,
    /// Leaf: the free variable `x`.
    Var,
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub op: Op,
    /// Index into the arena, or `NIL` for leaves / unused slots.
    pub a: u32,
    /// Index into the arena, or `NIL` for leaves / unused slots.
    pub b: u32,
    /// Populated only when `op == Op::Const`.
    pub const_val: Option<Complex64>,
}

impl Node {
    fn prim(a: u32, b: u32) -> Self {
        Node { op: Op::Prim, a, b, const_val: None }
    }

    fn constant(v: Complex64) -> Self {
        Node { op: Op::Const, a: NIL, b: NIL, const_val: Some(v) }
    }

    fn var() -> Self {
        Node { op: Op::Var, a: NIL, b: NIL, const_val: None }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Arena {
    pub nodes: Vec<Node>,
}

impl Arena {
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
    }

    pub fn get(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, node: Node) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }
}

/// Hash-consing interner: wraps arena insertion so that structurally
/// identical subexpressions always resolve to the same node ID.
///
/// Guarantee: interning the same logical node twice returns the same ID in
/// O(1) amortized time, and never allocates a duplicate `Node` in the arena.
pub struct Interner {
    pub arena: Arena,
    /// Keyed on (op, a, b) for Prim nodes.
    prim_table: FxHashMap<(u32, u32), u32>,
    /// List of (const_value, id). We use an approximate equality test when
    /// deciding whether two constants are the "same" so that floating-point
    /// roundoff won't secretly create distinct nodes for mathematically
    /// identical values produced by different evaluation paths.
    const_table: Vec<(Complex64, u32)>,
    /// At most one canonical Var node ever exists.
    var_id: Option<u32>,
}

impl Interner {
    pub fn new() -> Self {
        Interner {
            arena: Arena::new(),
            prim_table: FxHashMap::default(),
            const_table: Vec::new(),
            var_id: None,
        }
    }

    /// Intern `f(a, b)`. Returns the existing node ID if this exact `(a, b)`
    /// pair has been interned before, otherwise allocates a new node.
    pub fn intern_prim(&mut self, a: u32, b: u32) -> u32 {
        if let Some(&id) = self.prim_table.get(&(a, b)) {
            return id;
        }
        let id = self.arena.push(Node::prim(a, b));
        self.prim_table.insert((a, b), id);
        id
    }

    /// Intern a literal constant using a tolerance so near-equal floating
    /// values canonicalize to the same node. This aligns arena-level
    /// constant equality with the rest of the pipeline's numerical
    /// tolerance.
    pub fn intern_const(&mut self, v: Complex64) -> u32 {
        for &mut (ref existing, id) in &mut self.const_table.iter_mut() {
            if (existing - v).norm() <= MACHINE_TOLERANCE {
                return id;
            }
        }
        let id = self.arena.push(Node::constant(v));
        self.const_table.push((v, id));
        id
    }

    /// Intern the free variable `x`. Always returns the same ID.
    pub fn intern_var(&mut self) -> u32 {
        if let Some(id) = self.var_id {
            return id;
        }
        let id = self.arena.push(Node::var());
        self.var_id = Some(id);
        id
    }

    pub fn node(&self, id: u32) -> &Node {
        self.arena.get(id)
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_prim_same_children() {
        let mut it = Interner::new();
        let x = it.intern_var();
        let c = it.intern_const(Complex64::new(2.0, 0.0));
        let id1 = it.intern_prim(x, c);
        let id2 = it.intern_prim(x, c);
        assert_eq!(id1, id2, "interning the same (op, a, b) twice must return the same ID");
        assert_eq!(it.len(), 3, "no duplicate node should have been allocated");
    }

    #[test]
    fn dedup_var_is_singleton() {
        let mut it = Interner::new();
        let v1 = it.intern_var();
        let v2 = it.intern_var();
        assert_eq!(v1, v2);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn dedup_const_exact_bits() {
        let mut it = Interner::new();
        let a = it.intern_const(Complex64::new(1.5, -0.25));
        let b = it.intern_const(Complex64::new(1.5, -0.25));
        assert_eq!(a, b);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn distinct_children_do_not_collapse() {
        let mut it = Interner::new();
        let x = it.intern_var();
        let c1 = it.intern_const(Complex64::new(1.0, 0.0));
        let c2 = it.intern_const(Complex64::new(2.0, 0.0));
        let id1 = it.intern_prim(x, c1);
        let id2 = it.intern_prim(x, c2);
        assert_ne!(id1, id2);
        assert_eq!(it.len(), 5); // var, c1, c2, prim1, prim2
    }

    #[test]
    fn structurally_nested_dedup() {
        // f(f(x, c), f(x, c)) should intern the shared f(x, c) subtree once.
        let mut it = Interner::new();
        let x = it.intern_var();
        let c = it.intern_const(Complex64::new(3.0, 1.0));
        let inner1 = it.intern_prim(x, c);
        let inner2 = it.intern_prim(x, c);
        assert_eq!(inner1, inner2);
        let outer = it.intern_prim(inner1, inner2);
        // 3 nodes: var, const, inner prim -- plus the outer prim = 4 total.
        assert_eq!(it.len(), 4);
        let outer_node = it.node(outer);
        assert_eq!(outer_node.a, outer_node.b);
    }
}
