package org.optimizationBenchmarking.utils.ml.fitting.impl.dels;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;

/**
 * This curve fitter uses a combination of least-squares solvers,
 * Differential Evolution, and Simplex Search to obtain high-quality
 * solutions.
 */
public final class DELSFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "Differential Evolution + Least-Squares + Simplex Fitter"; //$NON-NLS-1$

  /** create */
  DELSFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _DELSFittingJob(builder);
  }

  /**
   * Get the globally shared instance of the DE/LS-based curve fitter
   *
   * @return the instance of the DE/LS-based curve fitter
   */
  public static final DELSFitter getInstance() {
    return _DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return DELSFitter.METHOD;
  }

  /** the instance holder */
  private static final class _DECurveFitterHolder {
    /** the shared instance */
    static final DELSFitter INSTANCE = new DELSFitter();
  }
}
