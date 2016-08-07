package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.Polynomials;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueCheckerMinMax;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplePermutationBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model of the type {@code a + exp(b + c*x)}. This model is a special
 * case of the {@link ExponentialDecayModel}. The
 * {@link ExponentialDecayModel} is defined as {@code a'+(b'*exp(c'*x^d'))}
 * . If {@code d'=1}, {@code c'=c}, {@code b'=exp(b)}, and {@code a=a'},
 * both models are identical.
 * </p>
 * <h2>Derivatives</h2>
 * <ol>
 * <li>{@code a}: {@code 1}</li>
 * <li>{@code b}: {@code exp(c*x+b)}</li>
 * <li>{@code c}: {@code x*exp(c*x+b)}</li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1</dt>
 * <dd>{@code a=(y1*exp(c*x0)-y0*exp(c*x1))/(exp(c*x0)-exp(c*x1))}</dd>
 * <dt>b.2.1</dt>
 * <dd>{@code b=log((y0-y1)/(exp(c*x0)-exp(c*x1)))}</dd>
 * <dt>b.2.2</dt>
 * <dd>{@code b=(x0*log(y1-a)-x1*log(y0-a))/(x0-x1)}</dd>
 * <dt>c.2.1</dt>
 * <dd>{@code c=(log(y0-a)-log(y1-a))/(x0-x1)}</dd>
 * </dl>
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=y0-exp(c*x0+b)}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=log(y0-a)-c*x0}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c=(log(y0-a)-b)/x0}</dd>
 * </dl>
 */
public class ExpLinearModel extends BasicModel {

  /** the checker for values of {@code b} */
  static final ParameterValueCheckerMinMax B = new ParameterValueCheckerMinMax(
      -1000d, 1000d);
  /** the checker for values of {@code c} */
  static final ParameterValueCheckerMinMax C = new ParameterValueCheckerMinMax(
      -1e10d, 1e10d);

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

  /** {@inheritDoc} */
  @Override
  public final IParameterGuesser createParameterGuesser(
      final IMatrix data) {
    return new __ExpLinearModelParameterGuesser(data);
  }

  /**
   * compute {@code a} based on two points and {@code c}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param x1
   *          the {@code x}-coordinate of the second point
   * @param y1
   *          the {@code y}-coordinate of the second point
   * @param c
   *          the {@code c} value
   * @return the guess for {@code a}
   */
  static final double _a_x0y0x1y1c(final double x0, final double y0,
      final double x1, final double y1, final double c) {
    final double cx1, cx2;
    cx1 = Math.exp(c * x0);
    cx2 = Math.exp(c * x1);
    return (((y1 * cx1) - (y0 * cx2)) / (cx1 - cx2));
  }

  /**
   * compute {@code b} based on two points and {@code c}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param x1
   *          the {@code x}-coordinate of the second point
   * @param y1
   *          the {@code y}-coordinate of the second point
   * @param c
   *          the {@code c} value
   * @return the guess for {@code b}
   */
  static final double _b_x0y0x1y1c(final double x0, final double y0,
      final double x1, final double y1, final double c) {
    return Math.log((y0 - y1) / (Math.exp(c * x0) - Math.exp(c * x1)));
  }

  /**
   * compute {@code b} based on two points and {@code a}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param x1
   *          the {@code x}-coordinate of the second point
   * @param y1
   *          the {@code y}-coordinate of the second point
   * @param a
   *          the {@code a} value
   * @return the guess for {@code b}
   */
  static final double _b_x0y0x1y1a(final double x0, final double y0,
      final double x1, final double y1, final double a) {
    return (((x0 * Math.log(y1 - a)) - (x1 * Math.log(y0 - a)))
        / (x0 - x1));
  }

  /**
   * compute {@code c} based on two points and {@code a}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param x1
   *          the {@code x}-coordinate of the second point
   * @param y1
   *          the {@code y}-coordinate of the second point
   * @param a
   *          the {@code a} value
   * @return the guess for {@code b}
   */
  static final double _c_x0y0x1y1a(final double x0, final double y0,
      final double x1, final double y1, final double a) {
    return ((Math.log(y0 - a) - Math.log(y1 - a)) / (x0 - x1));
  }

  /**
   * compute {@code a} based on one point and {@code b}, {@code c}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param b
   *          the {@code b} value
   * @param c
   *          the {@code c} value
   * @return the guess for {@code a}
   */
  static final double _a_x0y0bc(final double x0, final double y0,
      final double b, final double c) {
    return y0 - Math.exp((c * x0) + b);
  }

  /**
   * compute {@code b} based on one point and {@code a}, {@code c}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the {@code a} value
   * @param c
   *          the {@code c} value
   * @return the guess for {@code b}
   */
  static final double _b_x0y0ac(final double x0, final double y0,
      final double a, final double c) {
    return Math.log(y0 - a) - (c * x0);
  }

  /**
   * compute {@code c} based on one point and {@code a}, {@code b}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the {@code a} value
   * @param b
   *          the {@code b} value
   * @return the guess for {@code b}
   */
  static final double _c_x0y0ab(final double x0, final double y0,
      final double a, final double b) {
    return (Math.log(y0 - a) - b) / x0;
  }

  /**
   * compute a fallback parameter set the destination array
   *
   * @param minY
   *          the min Y
   * @param dest
   *          the destination
   * @param random
   *          the random number generator
   */
  static final void _fallback(final double minY, final double[] dest,
      final Random random) {
    dest[0] = (Math.abs(random.nextGaussian()) - minY);
    dest[1] = Math.abs(random.nextInt(10) * random.nextGaussian());
    dest[2] = (random.nextGaussian() - 1d);
  }

  /** the parameter guesser */
  private final class __ExpLinearModelParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /** the parameter value checker */
    private final ParameterValueCheckerMinMax m_A;

    /**
     * Create the parameter guesser
     *
     * @param data
     *          the data
     */
    __ExpLinearModelParameterGuesser(final IMatrix data) {
      super(data, 3, 2);

      final double minA;
      minA = (-this.m_minY);
      this.m_A = new ParameterValueCheckerMinMax(minA,
          Math.max(1e100d, Math.nextUp(minA)));
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExpLinearModel.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      ExpLinearModel._fallback(this.m_minY, dest, random);
    }

    /** {@inheritDoc} */
    @Override
    protected final void guessBasedOnPermutation(final double[] points,
        final double[] bestGuess, final double[] destGuess) {
      final double oldA, oldB, oldC;
      double[] temp;
      double newA, newB, newC;
      boolean changed, hasA, hasB, hasC;

      hasA = hasB = hasC = false;
      oldA = newA = bestGuess[0];
      oldB = newB = bestGuess[1];
      oldC = newC = bestGuess[2];

      do {
        changed = false;

        // find A
        findA: {
          if (!hasA) {

            newA = ExpLinearModel._a_x0y0x1y1c(points[0], points[1],
                points[2], points[2], (hasC ? newC : oldC));
            if (this.m_A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            newA = ExpLinearModel._a_x0y0bc(points[0], points[1],
                (hasB ? newB : oldB), (hasC ? newC : oldC));
            if (this.m_A.check(newA)) {
              changed = hasA = true;
              break findA;
            }
          }
        }

        // find B
        findB: {
          if (!hasB) {
            newB = ExpLinearModel._b_x0y0x1y1a(points[0], points[1],
                points[2], points[3], (hasA ? newA : oldA));
            if (ExpLinearModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = ExpLinearModel._b_x0y0x1y1c(points[0], points[1],
                points[2], points[3], (hasC ? newC : oldC));
            if (ExpLinearModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = ExpLinearModel._b_x0y0ac(points[0], points[1],
                (hasA ? newA : oldA), (hasC ? newC : oldC));
            if (ExpLinearModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }
          }
        }

        // find C
        findC: {
          if (!hasC) {
            newC = ExpLinearModel._c_x0y0x1y1a(points[0], points[1],
                points[2], points[3], (hasA ? newA : oldA));
            if (ExpLinearModel.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }

            newC = ExpLinearModel._c_x0y0ab(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB));
            if (ExpLinearModel.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }
          }
        }

        // OK, everything else has failed us
        emergency: {
          if (!(changed)) {

            if (!(hasB && hasC)) {
              temp = new double[2];
              if (Polynomials.degree1FindCoefficients(points[0],
                  Math.log(points[1] - (hasA ? newA : oldA)), points[2],
                  Math.log(points[3] - (hasA ? newA : oldA)),
                  temp) < Double.POSITIVE_INFINITY) {
                if ((!hasB) && ExpLinearModel.B.check(newB)) {
                  newB = temp[0];
                  hasB = changed = true;
                }
                if ((!hasC) && ExpLinearModel.C.check(newC)) {
                  newC = temp[1];
                  hasC = changed = true;
                }
                break emergency;
              }
            }

            if (!hasA) {
              newA = Math.nextUp(-this.m_minY);
              if (this.m_A.check(newA)) {
                hasA = changed = true;
              }
              break emergency;
            }
          }
        }
      } while (changed);

      destGuess[0] = newA;
      destGuess[1] = newB;
      destGuess[2] = newC;
    }
  }
}
