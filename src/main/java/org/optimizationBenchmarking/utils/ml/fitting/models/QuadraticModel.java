package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.Polynomials;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SampleBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * A polynomial of degree 2 to be fitted in order to model the relationship
 * of similarly-typed dimensions (time-time, objective-objective):
 * {@code a+b*x+c*x*x}.
 */
public final class QuadraticModel extends BasicModel {

  /** create */
  public QuadraticModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    return Polynomials.degree2Compute(x, parameters[0], parameters[1],
        parameters[2]);
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    Polynomials.degree2Gradient(x, gradient);
  }

  /** {@inheritDoc} */
  @Override
  public final int getParameterCount() {
    return 3;
  }

  /** {@inheritDoc} */
  @Override
  public final IParameterGuesser createParameterGuesser(
      final IMatrix data) {
    return new __PolynomialGuesser(data);
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final ITextOutput out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    renderer.renderParameter(0, out);
    out.append('+');
    renderer.renderParameter(1, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('+');
    renderer.renderParameter(2, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('\u00b2');
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    try (final IMath add = out.add()) {
      renderer.renderParameter(0, add);

      try (final IMath add2 = add.add()) {
        try (final IMath mul = add2.mul()) {
          renderer.renderParameter(1, mul);
          x.mathRender(mul, renderer);
        }

        try (final IMath mul = add2.mul()) {
          renderer.renderParameter(2, mul);
          try (final IMath sqr = mul.sqr()) {
            x.mathRender(sqr, renderer);
          }
        }
      }
    }
  }

  /** A parameter guesser for quadratic polynomials */
  private static final class __PolynomialGuesser
      extends SampleBasedParameterGuesser {

    /**
     * create the guesser
     *
     * @param data
     *          the data
     */
    __PolynomialGuesser(final IMatrix data) {
      super(data, 3);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean guess(final int variant, final double[] points,
        final double[] dest, final Random random) {
      switch (points.length) {
        case 2: {
          dest[1] = dest[2] = 0d;
          return (Polynomials.degree0FindCoefficients(points[0], points[1],
              dest) < Double.POSITIVE_INFINITY);
        }
        case 4: {
          dest[2] = 0d;
          return (Polynomials.degree1FindCoefficients(points[0], points[1],
              points[2], points[3], dest) < Double.POSITIVE_INFINITY);
        }
        case 6: {
          return (Polynomials.degree2FindCoefficients(points[0], points[1],
              points[2], points[3], points[4], points[5],
              dest) < Double.POSITIVE_INFINITY);
        }
        default: {
          this.fallback(0, dest, random);
          return true;
        }
      }
    }
  }
}
