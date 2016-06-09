package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;

/**
 * <p>
 * This curve fitter uses a combination of least-squares solvers and
 * simplex search and BOBYQA to fit a function.</p
 * <p>
 * It tends to be slower than
 * {@link org.optimizationBenchmarking.utils.ml.fitting.impl.dels.DELSFitter}
 * , but outperforms it in 66% of the runs (while losing in 33% of them).
 * </p>
 */
public final class LSSimplexFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "Least-Squares + Simplex + BOBYQA Fitter"; //$NON-NLS-1$

  /** create */
  LSSimplexFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _LSSimplexFittingJob(builder);
  }

  /**
   * Get the globally shared instance of the Opti-based curve fitter
   *
   * @return the instance of the Opti-based curve fitter
   */
  public static final LSSimplexFitter getInstance() {
    return _DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return LSSimplexFitter.METHOD;
  }

  /** the instance holder */
  private static final class _DECurveFitterHolder {
    /** the shared instance */
    static final LSSimplexFitter INSTANCE = new LSSimplexFitter();
  }
}
