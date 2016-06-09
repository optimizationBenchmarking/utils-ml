package org.optimizationBenchmarking.utils.ml.fitting.impl.dels;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;

/**
 * <p>
 * This curve fitter uses a combination of least-squares solvers,
 * Differential Evolution, and Simplex Search to obtain high-quality
 * solutions.
 * </p>
 * <p>
 * It tends to be faster than
 * {@link org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex.LSSimplexFitter}
 * , but its solution quality is often worse: In the examples,
 * {@link org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex.LSSimplexFitter}
 * seemingly outperforms it in 66% of the runs, while it is better in 33%
 * of them.
 * </p>
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
   * Get the globally shared instance of the Opti-based curve fitter
   *
   * @return the instance of the Opti-based curve fitter
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
