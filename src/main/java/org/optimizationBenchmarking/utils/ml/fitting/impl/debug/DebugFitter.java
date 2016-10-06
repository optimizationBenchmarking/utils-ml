package org.optimizationBenchmarking.utils.ml.fitting.impl.debug;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * This curve fitter uses Differential Evolution in a very fast way, mainly
 * for debugging purposes.
 */
public final class DebugFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "Differential Evolution"; //$NON-NLS-1$

  /** create */
  DebugFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _DebugFittingJob(builder);
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    textOut.append("Differential Evolution"); //$NON-NLS-1$
    return textCase.nextCase();
  }

  /**
   * Get the globally shared instance of the DE/LS-based curve fitter
   *
   * @return the instance of the DE/LS-based curve fitter
   */
  public static final DebugFitter getInstance() {
    return __DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return DebugFitter.METHOD;
  }

  /** the instance holder */
  private static final class __DECurveFitterHolder {
    /** the shared instance */
    static final DebugFitter INSTANCE = new DebugFitter();
  }
}
