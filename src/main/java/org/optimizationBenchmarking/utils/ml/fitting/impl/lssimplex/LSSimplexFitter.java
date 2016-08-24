package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * This curve fitter uses a combination of least-squares solvers and
 * simplex search and BOBYQA to fit a function.
 * </p>
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
   * Get the globally shared instance of the LS-Simplex-based curve fitter
   *
   * @return the instance of the LS-Simplex-based curve fitter
   */
  public static final LSSimplexFitter getInstance() {
    return __LSSimplexCurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;

    next = OptimizationBasedFitter.printLevenbergMarcquardt(textCase,
        textOut, true);
    textOut.append('-');
    next = OptimizationBasedFitter.printNelderMead(next, textOut, true);
    textOut.append('-');
    next = OptimizationBasedFitter.printBOBYQA(next, textOut, true);
    textOut.append(' ');
    return next.appendWord("hybrid", textOut); //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return LSSimplexFitter.METHOD;
  }

  /** the instance holder */
  private static final class __LSSimplexCurveFitterHolder {
    /** the shared instance */
    static final LSSimplexFitter INSTANCE = new LSSimplexFitter();
  }
}
