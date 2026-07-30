# expr-minimizer

Generic arena-based expression rewriter with equality saturation and a
stochastic fallback, built exactly to the task spec (Tasks 1-5).

## Layout

| File | Task | Contents |
|---|---|---|
| `src/arena.rs` | 1, 2 | `Op`, `Node`, `Arena`, `Interner` (hash-consing) + dedup unit tests |
| `src/fingerprint.rs` | 3 | extended-precision `evaluate` (via `rug`), `fingerprint` (xxHash3), `FingerprintIndex` + bucket unit tests |
| `src/saturate.rs` | 4 | `egg` `Language` (`L`), arena<->`RecExpr` translation, `OpCountCost`/`ByteSizeCost`, `minimize` |
| `src/rules.rs` | 4 | rule set as data (`Vec<Rewrite<L, ()>>`) — currently **placeholder rules only**, see below |
| `src/stochastic.rs` | 5 | population/mutation/crossover GP-style search, parallel fitness eval via `rayon` |
| `src/main.rs` | deliverable 5 | `minimize_expr(&Arena, u32, Duration) -> (Arena, u32, u64)` wiring saturation + fallback |

## Design decisions worth flagging

- **Constants in the `egg` language.** `egg::define_language!` requires leaf
  data to implement `Ord` (among other traits) for its e-class canonicalization,
  and `Complex64` doesn't implement `Ord`. So `saturate::L::Const` holds a
  `ConstId(u32)` — an index into a side `ConstTable` — rather than the raw
  complex value. `arena_to_recexpr`/`recexpr_to_arena` thread the table
  through the round trip. If you'd rather have the language own the value
  directly, an alternative is a newtype wrapping `(u64, u64)` bit patterns
  with a manual total-order `Ord` impl — happy to swap it if you prefer that
  over the indirection table.

- **Rule set is genuinely a stub.** Per the spec ("I'll supply the full rule
  list separately"), `rules.rs` only contains 2-3 placeholder rules to prove
  the data-driven wiring works — they are explicitly *not* validated
  algebraic identities (one of them, `placeholder-commute`, is almost
  certainly false for `f(a,b) = exp(a) - log(b)` and exists purely to
  exercise a bidirectional rewrite through the harness). Drop the real rules
  into `placeholder_rules()` (or point that function at a config loader)
  without touching `saturate.rs`.

- **`minimize_expr` budget split.** 70% of the budget goes to equality
  saturation; if `egg`'s `Runner` reports `StopReason::Saturated` (i.e. it
  fully explored the equivalence class, not just ran out of time), that
  result is trusted as-is and the stochastic fallback is skipped entirely.
  Otherwise the remaining budget goes to `stochastic::search`, seeded from
  whatever saturation already found, targeting the *original* expression's
  sample-point bytes so functional equivalence is preserved end-to-end.
  This 70/30 split is a reasonable-default guess, not something specified —
  easy to make configurable if you want it exposed.

- **Stochastic search parallelism.** Fitness evaluation (the expensive,
  extended-precision part) is parallelized with `rayon::par_iter`. Mutation
  and crossover run single-threaded because they call `Interner::intern_*`,
  which needs `&mut Interner` — true concurrent mutation of a shared
  hash-consing table would need a concurrent map (e.g. `dashmap`) or
  per-thread arenas merged after the fact, which felt like scope creep
  beyond what the spec asked for. Flag if you actually need that.

- **Fingerprint fixed-width encoding.** `evaluate` computes at
  extended precision (192 bits, ~58 decimal digits, comfortably over the
  "≥50 decimal digits" spec) so that two *mathematically* equal expressions
  don't diverge into different f64 roundings before hashing. The fingerprint
  itself then truncates to f64 bit patterns for a fixed-width, hashable byte
  buffer — the extended precision's job is to make that truncation reliable,
  not to be part of the hash itself.

## Not verified by compilation

I don't have network/cargo access in this environment, so this hasn't been
run through `cargo check`/`cargo test` against the real crates. I wrote it
carefully against the documented APIs for `egg` 0.9.x, `rug` 1.24.x
(with the `complex` feature explicitly enabled — it's not in `rug`'s
default feature set), `rustc-hash`, `rayon`, and `xxhash-rust`, but you
should expect to fix a handful of small things on first build — likely
candidates:

- exact method names on `rug::Complex` for `exp`/`ln` (I used
  `.exp()`/`.ln()` consuming `self`; some `rug` versions want
  `Complex::with_val(prec, z.exp_ref())` style instead)
- `egg::StopReason` variant naming/visibility across 0.9.x point releases
- `Id` <-> `usize` conversion ergonomics (`usize::from(id)` vs `id.into()`)

Run `cargo build` and `cargo test` locally; the unit tests in each module
(dedup in `arena.rs`, bucket collisions in `fingerprint.rs`, round-trip in
`saturate.rs`, termination in `stochastic.rs`, functional-equivalence in
`main.rs`) are there specifically so a first-build pass will surface any API
drift quickly.
