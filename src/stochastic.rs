//! Fallback stochastic search over the flat opcode arena, used when
//! equality saturation (`saturate.rs`) doesn't converge within its time
//! budget.
//!
//! Population = candidate expressions, each represented as a root `u32`
//! into a *shared* `Interner`/`Arena` (so structurally identical mutants
//! are automatically deduplicated by hash-consing). Mutation rewires a
//! randomly chosen `Prim` node's `(a, b)` children to a different existing
//! pair of arena node IDs, which is well-typed by construction since every
//! node in this system accepts arbitrary complex-valued children.
//!
//! Fitness evaluation (`sample_bytes`) is parallelized with `rayon`.
//! Mutation/crossover itself runs single-threaded because it interns new
//! nodes into the shared `Interner`, which requires mutable access; only
//! the (comparatively expensive, extended-precision) fitness evaluation is
//! parallelized across the population.

use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

use crate::arena::{Interner, Op as ArenaOp, NIL};
use crate::fingerprint::sample_bytes;

pub struct SearchConfig {
    pub population_size: usize,
    pub elite_count: usize,
    /// Weight on op-count in the fitness function.
    pub lambda: f64,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            population_size: 200,
            elite_count: 20,
            lambda: 0.05,
            mutation_rate: 0.7,
            crossover_rate: 0.3,
        }
    }
}

/// Total primitive-op count if the DAG were expanded into a tree (i.e. each
/// reference to a shared subexpression counts separately). This matches the
/// `OpCountCost` extraction metric used in `saturate.rs`.
pub fn op_count(interner: &Interner, node: u32) -> u64 {
    match interner.node(node).op {
        ArenaOp::Var | ArenaOp::Const => 1,
        ArenaOp::Prim => {
            let n = interner.node(node);
            1 + op_count(interner, n.a) + op_count(interner, n.b)
        }
    }
}

/// Hamming distance between two equal-length byte arrays (the raw,
/// pre-hash sample-point outputs), used instead of the collapsed
/// fingerprint hash so the fitness landscape has a smooth gradient rather
/// than an all-or-nothing step function.
fn fingerprint_distance(a: &[u8], b: &[u8]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u64
}

fn fitness(interner: &Interner, candidate: u32, target_bytes: &[u8], lambda: f64) -> f64 {
    let bytes = sample_bytes(interner, candidate);
    let dist = fingerprint_distance(&bytes, target_bytes) as f64;
    let cost = op_count(interner, candidate) as f64;
    -dist - lambda * cost
}

/// Collect all node IDs reachable from `root` (including `root` itself),
/// deduplicated. Used to draw "existing arena node IDs" for mutation.
fn subtree_nodes(interner: &Interner, root: u32) -> Vec<u32> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        out.push(id);
        let n = interner.node(id);
        if n.a != NIL {
            stack.push(n.a);
        }
        if n.b != NIL {
            stack.push(n.b);
        }
    }
    out
}

/// Random root-to-node path: at each `Prim` node, flip a coin to either
/// stop there or descend into a randomly chosen child, recording the
/// direction taken (`true` = went into `a`, `false` = went into `b`).
/// Leaves always stop. Returns `(path, target)` where `path` is the
/// sequence of `(ancestor_id, direction_taken)` pairs from `root` down to
/// (but not including) `target`.
fn random_path(interner: &Interner, root: u32, rng: &mut impl Rng) -> (Vec<(u32, bool)>, u32) {
    let mut path = Vec::new();
    let mut current = root;
    loop {
        let n = interner.node(current);
        match n.op {
            ArenaOp::Var | ArenaOp::Const => return (path, current),
            ArenaOp::Prim => {
                if rng.gen_bool(0.5) {
                    return (path, current);
                }
                let go_left = rng.gen_bool(0.5);
                if go_left {
                    path.push((current, true));
                    current = n.a;
                } else {
                    path.push((current, false));
                    current = n.b;
                }
            }
        }
    }
}

/// Rebuild the chain of ancestors from a mutated/replaced `target` back up
/// to a new root, re-interning each ancestor along `path` (traversed in
/// reverse) with its updated child.
fn rebuild(interner: &mut Interner, path: &[(u32, bool)], mut current: u32) -> u32 {
    for &(parent_id, went_left) in path.iter().rev() {
        let parent = *interner.node(parent_id);
        let (a, b) = if went_left { (current, parent.b) } else { (parent.a, current) };
        current = interner.intern_prim(a, b);
    }
    current
}

/// Mutate `root`: pick a random position via `random_path`, then replace
/// that position with a fresh `Prim` node whose two children are drawn
/// uniformly from the existing arena node IDs (well-typed automatically,
/// since every node accepts arbitrary complex-valued children).
fn mutate(interner: &mut Interner, root: u32, rng: &mut impl Rng) -> u32 {
    if interner.len() == 0 {
        return root;
    }
    let (path, _target) = random_path(interner, root, rng);
    let max_id = interner.len() as u32;
    let a = rng.gen_range(0..max_id);
    let b = rng.gen_range(0..max_id);
    let replacement = interner.intern_prim(a, b);
    rebuild(interner, &path, replacement)
}

/// Crossover: splice a randomly chosen subtree of `donor` into a randomly
/// chosen position of `recipient`.
fn crossover(interner: &mut Interner, recipient: u32, donor: u32, rng: &mut impl Rng) -> u32 {
    let (path, _target) = random_path(interner, recipient, rng);
    let donor_nodes = subtree_nodes(interner, donor);
    let replacement = *donor_nodes.choose(rng).unwrap_or(&donor);
    rebuild(interner, &path, replacement)
}

pub struct SearchResult {
    pub best: u32,
    pub best_fitness: f64,
    pub generations_run: u64,
    pub exact_match_found: bool,
}

/// Standard generational loop: evaluate fitness (in parallel across the
/// population), select the top-k elites, refill the rest of the population
/// via mutation/crossover of the elites, repeat until `budget` is exhausted
/// or a fingerprint-exact match (distance 0) is found.
///
/// `seed` is the starting candidate (typically the equality-saturation
/// output, or the original expression, cast into the shared `interner`).
pub fn search(
    interner: &mut Interner,
    seed: u32,
    target_bytes: &[u8],
    config: &SearchConfig,
    budget: Duration,
) -> SearchResult {
    let mut rng = rand::thread_rng();
    let start = Instant::now();

    // Seed the initial population with mutants of `seed` so the search
    // starts near a known-valid point rather than from scratch.
    let mut population: Vec<u32> = (0..config.population_size)
        .map(|i| if i == 0 { seed } else { mutate(interner, seed, &mut rng) })
        .collect();

    let mut best = seed;
    let mut best_fitness = fitness(interner, seed, target_bytes, config.lambda);
    let mut generations_run = 0u64;
    let mut exact_match_found = false;

    while start.elapsed() < budget {
        // Fitness evaluation is read-only over `interner`, so this part is
        // safely parallelized across the population.
        let scored: Vec<(u32, f64)> = population
            .par_iter()
            .map(|&cand| (cand, fitness(interner, cand, target_bytes, config.lambda)))
            .collect();

        let mut ranked = scored;
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&(top_id, top_fit)) = ranked.first() {
            if top_fit > best_fitness {
                best = top_id;
                best_fitness = top_fit;
            }
            let bytes = sample_bytes(interner, top_id);
            if fingerprint_distance(&bytes, target_bytes) == 0 {
                exact_match_found = true;
                if top_fit > best_fitness || best == seed {
                    best = top_id;
                    best_fitness = top_fit;
                }
                break;
            }
        }

        let elites: Vec<u32> = ranked
            .iter()
            .take(config.elite_count.max(1))
            .map(|&(id, _)| id)
            .collect();

        // Refill the next generation from the elites via mutation/crossover
        // (sequential: each intern_prim call needs &mut Interner).
        let mut next_gen = elites.clone();
        while next_gen.len() < config.population_size {
            let parent_a = *elites.choose(&mut rng).unwrap();
            let roll: f64 = rng.gen();
            let child = if roll < config.crossover_rate {
                let parent_b = *elites.choose(&mut rng).unwrap();
                crossover(interner, parent_a, parent_b, &mut rng)
            } else if roll < config.crossover_rate + config.mutation_rate {
                mutate(interner, parent_a, &mut rng)
            } else {
                parent_a
            };
            next_gen.push(child);
        }
        population = next_gen;
        generations_run += 1;
    }

    SearchResult { best, best_fitness, generations_run, exact_match_found }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Interner;
    use num_complex::Complex64;

    #[test]
    fn search_runs_and_terminates_within_budget() {
        let mut interner = Interner::new();
        let x = interner.intern_var();
        let c = interner.intern_const(Complex64::new(1.0, 0.0));
        let seed = interner.intern_prim(x, c);
        let target_bytes = sample_bytes(&interner, seed);

        let config = SearchConfig { population_size: 20, elite_count: 4, ..Default::default() };
        let result = search(&mut interner, seed, &target_bytes, &config, Duration::from_millis(300));

        // The seed itself is a perfect match, so distance-0 should be found
        // essentially immediately (population includes the seed at index 0).
        assert!(result.exact_match_found);
    }

    #[test]
    fn op_count_matches_tree_expansion() {
        let mut interner = Interner::new();
        let x = interner.intern_var();
        let c = interner.intern_const(Complex64::new(2.0, 0.0));
        let inner = interner.intern_prim(x, c);
        let outer = interner.intern_prim(inner, inner); // shared child, DAG
        // Tree-expansion count: outer(1) + inner(3) + inner(3) = 7.
        assert_eq!(op_count(&interner, outer), 7);
    }
}
