package org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;

/**
 * This curve fitter uses a combination of least-squares solvers and CMA-ES
 * to obtain high-quality solutions.
 */
public final class CMAESLSFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "CMA-ES + Least-Squares Fitter"; //$NON-NLS-1$

  /** create */
  CMAESLSFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _CMAESLSFittingJob(builder);
  }

  /**
   * Get the globally shared instance of the Opti-based curve fitter
   *
   * @return the instance of the Opti-based curve fitter
   */
  public static final CMAESLSFitter getInstance() {
    return _DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return CMAESLSFitter.METHOD;
  }

  /** the instance holder */
  private static final class _DECurveFitterHolder {
    /** the shared instance */
    static final CMAESLSFitter INSTANCE = new CMAESLSFitter();
  }
}
