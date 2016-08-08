package org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

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
   * Get the globally shared instance of the CMA-ES/LS-based curve fitter
   *
   * @return the instance of the CMA-ES/LS-based curve fitter
   */
  public static final CMAESLSFitter getInstance() {
    return _DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;

    next = OptimizationBasedFitter.printCMAES(textCase, textOut, true);
    textOut.append('-');
    next = OptimizationBasedFitter.printLevenbergMarcquardt(next, textOut,
        false);
    textOut.append(' ');
    return next.appendWord("hybrid", textOut); //$NON-NLS-1$
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
