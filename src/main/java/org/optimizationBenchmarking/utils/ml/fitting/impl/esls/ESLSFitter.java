package org.optimizationBenchmarking.utils.ml.fitting.impl.esls;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;

/**
 * This curve fitter uses a combination of least-squares solvers and a
 * (mu+lambda)-Evolution Strategy to obtain high-quality solutions.
 */
public final class ESLSFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "ES + Least-Squares Fitter"; //$NON-NLS-1$

  /** create */
  ESLSFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _ESLSFittingJob(builder);
  }

  /**
   * Get the globally shared instance of the ES/LS-based curve fitter
   *
   * @return the instance of the ES/LS-based curve fitter
   */
  public static final ESLSFitter getInstance() {
    return _DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return ESLSFitter.METHOD;
  }

  /** the instance holder */
  private static final class _DECurveFitterHolder {
    /** the shared instance */
    static final ESLSFitter INSTANCE = new ESLSFitter();
  }
}
