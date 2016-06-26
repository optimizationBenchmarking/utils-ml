package org.optimizationBenchmarking.utils.ml.fitting.models;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model of the type {@code a + exp(b + c*log(x+d))}.
 * </p>
 * <h2>Derivatives</h2>
 * <ol>
 * <li>{@code a}: {@code 1}</li>
 * <li>{@code b}: {@code exp(b)*(d+x)^c}</li>
 * <li>{@code c}: {@code exp(b)*(d+x)^c*log(d+x)}</li>
 * <li>{@code d}: {@code exp(b)*c*(d+x)^(c-1)}</li>
 * </ol>
 */
public class ExpLinearModelOverLogX extends BasicModel {
  /** create */
  public ExpLinearModelOverLogX() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    return (parameters[0] + Math.exp(
        parameters[1] + (parameters[2] * Math.log(parameters[3] + x))));
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double expb, dpx, dfdb, c;

    parameters[0] = 1d;// a
    expb = Math.exp(parameters[1]);
    dpx = parameters[3] + x;
    c = parameters[2];
    dfdb = expb * Math.pow(dpx, c);
    parameters[1] = dfdb;// b
    parameters[2] = dfdb * Math.log(dpx);// c
    parameters[2] = expb * c * Math.pow(dpx, (c - 1d));// d
  }

  /** {@inheritDoc} */
  @Override
  public final int getParameterCount() {
    return 4;
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
    out.append("*log(");//$NON-NLS-1$
    renderer.renderParameter(3, out);
    out.append('+');
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
              try (final IMath log = mul.log()) {
                try (final IMath braces2 = log.inBraces()) {
                  try (final IMath add3 = braces2.add()) {
                    renderer.renderParameter(3, add3);
                    x.mathRender(add3, renderer);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}