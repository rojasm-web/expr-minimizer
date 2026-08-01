/// Numeric tolerance used for deciding when two complex values are "the
/// same" across the crate. This single source-of-truth keeps arena-level
/// interning, e-graph folding, and other numeric comparisons consistent.
pub const MACHINE_TOLERANCE: f64 = 1e-12;

/// Default x-range used by the plotting workflow.
pub const PLOT_XMIN: f64 = -1000000000000.0;
pub const PLOT_XMAX: f64 = 1000000000000.0;

/// Base window size used for the rolling-average smoothing curve.
pub const PLOT_SMOOTHING_WINDOW: f64 = 0.01;

/// How strongly the smoothing window grows with distance from zero.
pub const PLOT_SMOOTHING_SCALE: f64 = 1.0;
