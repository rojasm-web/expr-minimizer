/// Numeric tolerance used for deciding when two complex values are "the
/// same" across the crate. This single source-of-truth keeps arena-level
/// interning, e-graph folding, and other numeric comparisons consistent.
pub const MACHINE_TOLERANCE: f64 = 1e-12;
