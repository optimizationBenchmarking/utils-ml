package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.Polynomials;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueChecker;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueCheckerMinMax;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplePermutationBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
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
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1</dt>
 * <dd>{@code a=(y0*(d+x1)^c-y1*(d+x0)^c)/((d+x1)^c-(d+x0)^c)}, requires
 * {@code b} to be imaginary or 0</dd>
 * <dt>b.2.1</dt>
 * <dd>
 * {@code b=-(log(y0-a)*log(d+x1)-log(y1-a)*log(d+x0))/(log(d+x0)-log(d+x1))}
 * </dd>
 * <dt>c.2.1</dt>
 * <dd>{@code c=-(log(y1-a)-log(y0-a))/(log(d+x0)-log(d+x1))}</dd>
 * </dl>
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=y0-exp(c*log(x0+d)+b)}</dd>
 * <dt>a.1.2</dt>
 * <dd>{@code a=y0-exp(b)*(d+x0)^c}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=log((y0-a)*(d+x0)^(-c))}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c=(log(exp(-b)*(y0-a)))/(log(d+x0))}</dd>
 * <dt>d.1.1</dt>
 * <dd>{@code d=(-exp(-b)*(a-y0))^(1/c)-x0}</dd>
 * </dl>
 */
public class ExpLinearModelOverLogX extends BasicModel {

  /** the checker for values of {@code b} */
  static final ParameterValueCheckerMinMax B = ExpLinearModel.B;
  /** the checker for values of {@code c} */
  static final ParameterValueCheckerMinMax C = ExpLinearModel.C;

  /** create */
  public ExpLinearModelOverLogX() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "exp-linear model over logarithmic inputs"; //$NON-NLS-1$
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
  @SuppressWarnings({ "resource", "null" })
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    final INegatableParameterRenderer negatableRenderer;
    final IMath closerC, closerD;
    final boolean negateC, negateD;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    try (final IMath add = out.add()) {
      renderer.renderParameter(0, add);
      try (final IMath exp = add.exp()) {
        try (final IMath braces = exp.inBraces()) {

          if ((negatableRenderer != null)
              && (negatableRenderer.isNegative(2))) {
            negateC = true;
            closerC = braces.sub();
          } else {
            negateC = false;
            closerC = braces.add();
          }

          try {
            renderer.renderParameter(1, closerC);

            try (final IMath mul = closerC.mul()) {

              if (negateC) {
                negatableRenderer.renderNegatedParameter(2, mul);
              } else {
                renderer.renderParameter(2, mul);
              }

              try (final IMath log = mul.ln()) {
                try (final IMath braces2 = log.inBraces()) {

                  if ((negatableRenderer != null)
                      && (negatableRenderer.isNegative(3))) {
                    negateD = true;
                    closerD = braces2.sub();
                  } else {
                    negateD = false;
                    closerD = braces2.add();
                  }

                  try {
                    x.mathRender(closerD, renderer);
                    if (negateD) {
                      negatableRenderer.renderNegatedParameter(3, closerD);
                    } else {
                      renderer.renderParameter(3, closerD);
                    }
                  } finally {
                    closerD.close();
                  }

                }
              }
            }

          } finally {
            closerC.close();
          }

        }
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public final IParameterGuesser createParameterGuesser(
      final IMatrix data) {
    return new __ExpLinearModelOverLogXParameterGuesser(data);
  }

  /**
   * compute {@code a} based on two points and {@code c} and {@code d}.
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
   * @param d
   *          the {@code d} value
   * @return the guess for {@code a}
   */
  static final double _a_x0y0x1y1cd(final double x0, final double y0,
      final double x1, final double y1, final double c, final double d) {
    final double dx1powc, dx0powc;
    dx1powc = Math.pow((d + x1), c);
    dx0powc = Math.pow((d + x0), c);
    return ((y0 * dx1powc) - (y1 * dx0powc)) / (dx1powc - dx0powc);
  }

  /**
   * compute {@code b} based on two points and {@code a} and {@code d}.
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
   * @param d
   *          the {@code d} value
   * @return the guess for {@code a}
   */
  static final double _b_x0y0x1y1ad(final double x0, final double y0,
      final double x1, final double y1, final double a, final double d) {
    final double logdx0, logdx1;

    logdx0 = Math.log(d + x0);
    logdx1 = Math.log(d + x1);
    return -((((Math.log(y0 - a)) * logdx1) - (Math.log(y1 - a) * logdx0))
        / (logdx0 - logdx1));
  }

  /**
   * compute {@code c} based on two points and {@code a} and {@code d}.
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
   * @param d
   *          the {@code d} value
   * @return the guess for {@code c}
   */
  static final double _c_x0y0x1y1ad(final double x0, final double y0,
      final double x1, final double y1, final double a, final double d) {
    return -(Math.log(y1 - a) - Math.log(y0 - a))
        / (Math.log(d + x0) - Math.log(d + x1));
  }

  /**
   * compute {@code a} based on two points and {@code b}, {@code c} and
   * {@code d}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param b
   *          the {@code b} value
   * @param c
   *          the {@code c} value
   * @param d
   *          the {@code d} value
   * @return the guess for {@code a}
   */
  static final double _a_x0y0bcd1(final double x0, final double y0,
      final double b, final double c, final double d) {
    return y0 - Math.exp((c * Math.log(x0 + d)) + b);
  }

  /**
   * compute {@code a} based on two points and {@code b}, {@code c} and
   * {@code d}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param b
   *          the {@code b} value
   * @param c
   *          the {@code c} value
   * @param d
   *          the {@code d} value
   * @return the guess for {@code a}
   */
  static final double _a_x0y0bcd2(final double x0, final double y0,
      final double b, final double c, final double d) {
    return y0 - (Math.exp(b) * Math.pow((d + x0), c));
  }

  /**
   * compute {@code b} based on two points and {@code a}, {@code c} and
   * {@code d}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the {@code a} value
   * @param c
   *          the {@code c} value
   * @param d
   *          the {@code d} value
   * @return the guess for {@code b}
   */
  static final double _b_x0y0acd(final double x0, final double y0,
      final double a, final double c, final double d) {
    return Math.log((y0 - a) * Math.pow((d + x0), (-c)));
  }

  /**
   * compute {@code c} based on two points and {@code a}, {@code b} and
   * {@code d}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the {@code a} value
   * @param b
   *          the {@code b} value
   * @param d
   *          the {@code d} value
   * @return the guess for {@code c}
   */
  static final double _c_x0y0abd(final double x0, final double y0,
      final double a, final double b, final double d) {
    return (Math.log(Math.exp(-b) * (y0 - a))) / (Math.log(d + x0));
  }

  /**
   * compute {@code d} based on two points and {@code a}, {@code b} and
   * {@code c}.
   *
   * @param x0
   *          the {@code x}-coordinate of the first point
   * @param y0
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the {@code a} value
   * @param b
   *          the {@code b} value
   * @param c
   *          the {@code c} value
   * @return the guess for {@code d}
   */
  static final double _d_x0y0abc(final double x0, final double y0,
      final double a, final double b, final double c) {
    return Math.pow((-Math.exp(-b) * (a - y0)), (1 / c)) - x0;
  }

  /** the parameter guesser */
  private final class __ExpLinearModelOverLogXParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /** the parameter value checker for {@code a} */
    private final ParameterValueCheckerMinMax m_A;

    /** the parameter value checker for {@code d} */
    private final ParameterValueCheckerMinMax m_D;

    /**
     * Create the parameter guesser
     *
     * @param data
     *          the data
     */
    __ExpLinearModelOverLogXParameterGuesser(final IMatrix data) {
      super(data, 3, 2);

      final double minA, minD;
      minA = (-this.m_minY);
      minD = (-this.m_minX);
      this.m_A = new ParameterValueCheckerMinMax(minA,
          Math.max(1e100d, Math.nextUp(minA)));
      this.m_D = new ParameterValueCheckerMinMax(minD,
          Math.max(1e100d, Math.nextUp(minD)));
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExpLinearModelOverLogX.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      ExpLinearModel._fallback(this.m_minY, dest, random);
      dest[3] = ((Math.abs(random.nextGaussian()) + 1d)
          * (random.nextDouble() - this.m_minX));
    }

    /** {@inheritDoc} */
    @Override
    protected final void guessBasedOnPermutation(final double[] points,
        final double[] bestGuess, final double[] destGuess) {
      final double oldA, oldB, oldC, oldD;
      double[] temp;
      double newA, newB, newC, newD;
      boolean changed, hasA, hasB, hasC, hasD;

      hasA = hasB = hasC = hasD = false;
      oldA = newA = bestGuess[0];
      oldB = newB = bestGuess[1];
      oldC = newC = bestGuess[2];
      oldD = newD = bestGuess[3];

      do {
        changed = false;

        // find A
        findA: {
          if (!hasA) {

            newA = ExpLinearModelOverLogX._a_x0y0x1y1cd(points[0],
                points[1], points[2], points[3], (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (this.m_A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            newA = ParameterValueChecker.choose(//
                ExpLinearModelOverLogX._a_x0y0bcd1(points[0], points[1],
                    (hasB ? newB : oldB), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX._a_x0y0bcd1(points[0], points[1],
                    (hasB ? newB : oldB), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                this.m_A);
            if (this.m_A.check(newA)) {
              changed = hasA = true;
              break findA;
            }
          }
        }

        // find B
        findB: {
          if (!hasB) {
            newB = ExpLinearModelOverLogX._b_x0y0x1y1ad(points[0],
                points[1], points[2], points[3], (hasA ? newA : oldA),
                (hasD ? newD : oldD));
            if (ExpLinearModelOverLogX.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = ExpLinearModelOverLogX._b_x0y0acd(points[0], points[1],
                (hasA ? newA : oldA), (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (ExpLinearModelOverLogX.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }
          }
        }

        // find C
        findC: {
          if (!hasC) {
            newC = ExpLinearModelOverLogX._c_x0y0x1y1ad(points[0],
                points[1], points[2], points[3], (hasA ? newA : oldA),
                (hasD ? newD : oldD));
            if (ExpLinearModelOverLogX.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }

            newC = ExpLinearModelOverLogX._c_x0y0abd(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasD ? newD : oldD));
            if (ExpLinearModel.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }
          }
        }

        // find D
        findC: {
          if (!hasD) {
            newD = ExpLinearModelOverLogX._d_x0y0abc(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasC ? newC : oldC));
            if (this.m_D.check(newD)) {
              changed = hasD = true;
              break findC;
            }
          }
        }

        // OK, everything else has failed us
        emergency: {
          if (!(changed)) {
            if (!(hasB && hasC)) {
              temp = new double[2];
              if (Polynomials.degree1FindCoefficients(//
                  Math.log(points[0] + (hasD ? newD : oldD)),
                  Math.log(points[1] - (hasA ? newA : oldA)), //
                  Math.log(points[2] + (hasD ? newD : oldD)),
                  Math.log(points[3] - (hasA ? newA : oldA)), //
                  temp) < Double.POSITIVE_INFINITY) {
                if ((!hasB) && ExpLinearModelOverLogX.B.check(newB)) {
                  newB = temp[0];
                  hasB = changed = true;
                }
                if ((!hasC) && ExpLinearModelOverLogX.C.check(newC)) {
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

            if (!hasD) {
              newD = Math.nextUp(-this.m_minX);
              if (this.m_D.check(newD)) {
                hasD = changed = true;
              }
              break emergency;
            }
          }
        }
      } while (changed);

      destGuess[0] = newA;
      destGuess[1] = newB;
      destGuess[2] = newC;
      destGuess[3] = newD;
    }
  }
}