package org.optimizationBenchmarking.utils.ml.fitting.models;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model of the type {@code a + exp(b + c*x)}.
 * </p>
 * <h2>Derivatives</h2>
 * <ol>
 * <li>{@code a}: {@code 1}</li>
 * <li>{@code b}: {@code exp(c*x+b)}</li>
 * <li>{@code c}: {@code x*exp(c*x+b)}</li>
 * </ol>
 */
public class ExpLinearModel extends BasicModel {

  /** create */
  public ExpLinearModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    return (parameters[0] + Math.exp(parameters[1] + (parameters[2] * x)));
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double exp;
    parameters[0] = 1d;// a
    exp = Math.exp(parameters[1] + (parameters[2] * x));
    parameters[1] = exp;// b
    parameters[2] = (x * exp);// c
  }

  /** {@inheritDoc} */
  @Override
  public final int getParameterCount() {
    return 3;
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final ITextOutput out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    renderer.renderParameter(0, out);
    out.append("+exp("); //$NON-NLS-1$
    renderer.renderParameter(1, out);
    out.append('+');
    renderer.renderParameter(2, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append(')');
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    try (final IMath add = out.add()) {
      renderer.renderParameter(0, add);
      try (final IMath exp = add.exp()) {
        try (final IMath braces = exp.inBraces()) {
          try (final IMath add2 = braces.add()) {
            renderer.renderParameter(1, add2);
            try (final IMath mul = add2.mul()) {
              renderer.renderParameter(2, mul);
              x.mathRender(mul, renderer);
            }
          }
        }
      }
    }
  }
}
