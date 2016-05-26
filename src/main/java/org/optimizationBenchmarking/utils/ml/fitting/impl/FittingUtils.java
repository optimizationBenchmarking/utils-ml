package org.optimizationBenchmarking.utils.ml.fitting.impl;

import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.math.text.ABCParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A fitting utility class. */
public final class FittingUtils {

  /** Fitting utilities */
  private FittingUtils() {
    ErrorUtils.doNotCall();
  }

  /**
   * Render a function to be fitted.
   *
   * @param function
   *          the function
   * @param textOut
   *          the text output to render to
   */
  public static final void renderFunctionToFit(
      final ParametricUnaryFunction function, final ITextOutput textOut) {
    textOut.append(function);
    textOut.append(' ');
    textOut.append('(');
    function.mathRender(textOut, ABCParameterRenderer.INSTANCE);
    textOut.append(')');
  }

  /**
   * Render a fitting result
   *
   * @param parameters
   *          the parameters
   * @param quality
   *          the fitting quality
   * @param textOut
   *          the text output to render to
   */
  public static final void renderFittingResult(final double[] parameters,
      final double quality, final ITextOutput textOut) {
    char separator;

    separator = '[';
    for (final double value : parameters) {
      textOut.append(separator);
      separator = ',';
      textOut.append(value);
    }
    textOut.append("] with quality ");//$NON-NLS-1$
    textOut.append(quality);
  }
}
