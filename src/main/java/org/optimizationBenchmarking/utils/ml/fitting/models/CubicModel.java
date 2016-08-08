package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.Polynomials;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SampleBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A model of the type {@code a + b*x + c*x^2 + d*x^3}. */
public class CubicModel extends BasicModel {

  /** create */
  public CubicModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    return Polynomials.degree3Compute(x, parameters[0], parameters[1],
        parameters[2], parameters[3]);
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    Polynomials.degree3Gradient(x, gradient);
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
    out.append('+');
    renderer.renderParameter(1, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('+');
    renderer.renderParameter(2, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('\u00b2');
    out.append('+');
    renderer.renderParameter(2, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('\u00b3');
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "resource", "null" })
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    final INegatableParameterRenderer negatableRenderer;
    final IMath closerA, closerB, closerC;
    final boolean negateA, negateB, negateC;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    if ((negatableRenderer != null) && (negatableRenderer.isNegative(0))) {
      negateA = true;
      closerA = out.sub();
    } else {
      negateA = false;
      closerA = out.add();
    }

    try {

      if ((negatableRenderer != null)
          && (negatableRenderer.isNegative(1))) {
        negateB = true;
        closerB = closerA.sub();
      } else {
        negateB = false;
        closerB = closerA.add();
      }

      try {

        if ((negatableRenderer != null)
            && (negatableRenderer.isNegative(2))) {
          negateC = true;
          closerC = closerB.sub();
        } else {
          negateC = false;
          closerC = closerB.add();
        }

        try {

          try (final IMath mulD = closerC.mul()) {
            renderer.renderParameter(3, mulD);
            try (final IMath cube = mulD.cube()) {
              x.mathRender(cube, renderer);
            }
          }

          try (final IMath mulC = closerC.mul()) {
            if (negateC) {
              negatableRenderer.renderNegatedParameter(2, mulC);
            } else {
              renderer.renderParameter(2, mulC);
            }
            try (final IMath cube = mulC.sqr()) {
              x.mathRender(cube, renderer);
            }
          }

        } finally {
          closerC.close();
        }

        try (final IMath mulB = closerB.mul()) {
          if (negateB) {
            negatableRenderer.renderNegatedParameter(1, mulB);
          } else {
            renderer.renderParameter(1, mulB);
          }
          x.mathRender(mulB, renderer);
        }

      } finally {
        closerB.close();
      }

      if (negateA) {
        negatableRenderer.renderNegatedParameter(0, closerA);
      } else {
        renderer.renderParameter(0, closerA);
      }

    } finally {
      closerA.close();
    }
  }

  /** {@inheritDoc} */
  @Override
  public final IParameterGuesser createParameterGuesser(
      final IMatrix data) {
    return new __PolynomialGuesser(data);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "cubic polynomial"; //$NON-NLS-1$
  }

  /** A parameter guesser for cubic polynomials */
  private static final class __PolynomialGuesser
      extends SampleBasedParameterGuesser {

    /**
     * create the guesser
     *
     * @param data
     *          the data
     */
    __PolynomialGuesser(final IMatrix data) {
      super(data, 4);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean guess(final double[] points,
        final double[] dest, final Random random) {
      return (Polynomials.degree3FindCoefficients(points[0], points[1],
          points[2], points[3], points[4], points[5], points[6], points[7],
          dest) < Double.POSITIVE_INFINITY);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean fallback(final double[] points,
        final double[] dest, final Random random) {
      switch (points.length) {
        case 2: {
          return (Polynomials.degree0FindCoefficients(points[0], points[1],
              dest) < Double.POSITIVE_INFINITY);
        }
        case 4: {
          return (Polynomials.degree1FindCoefficients(points[0], points[1],
              points[2], points[3], dest) < Double.POSITIVE_INFINITY);
        }
        case 6: {
          return (Polynomials.degree2FindCoefficients(points[0], points[1],
              points[2], points[3], points[4], points[5],
              dest) < Double.POSITIVE_INFINITY);
        }
        default: {
          this.fallback(dest, random);
          return true;
        }
      }
    }
  }
}
