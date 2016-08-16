package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueChecker;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueCheckerMinMax;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplePermutationBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model of the type {@code a + b*exp(c*log(x+d))}.
 * </p>
 * <h2>Derivatives</h2>
 * <ol>
 * <li>{@code a}: {@code 1}</li>
 * <li>{@code b}: {@code (d+x)^c}</li>
 * <li>{@code c}: {@code b*(d+x)^c*log(x+d)}</li>
 * <li>{@code d}: {@code b*c*(d+x)^(c-1)}</li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1</dt>
 * <dd>
 * {@code a=(exp(c*log(x1+d))*y0-exp(c*log(x0+d))*y1)/(exp(c*log(x1+d))-exp(c*log(x0+d)))}
 * </dd>
 * <dt>a.2.2</dt>
 * <dd>{@code a=(y1*(d+x0)^c-y0*(d+x1)^c)/((d+x0)^c-(d+x1)^c)}</dd>
 * <dt>b.2.1</dt>
 * <dd>{@code b=(y1-y0)/(exp(c*log(x1+d))-exp(c*log(x0+d)))}</dd>
 * <dt>b.2.2</dt>
 * <dd>{@code b=(y0-y1)/((d+x0)^c-(d+x1)^c)}</dd>
 * <dt>c.2.1</dt>
 * <dd>{@code c=-(log(y1-a)-log(y0-a))/(log(d+x0)-log(d+x1))}</dd>
 * </dl>
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=y1-b*exp(c*log(x1+d))}</dd>
 * <dt>a.1.2</dt>
 * <dd>{@code a=y1-b*(d+x1)^c}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=exp(-(c*log(x1+d)))*(y1-a)}</dd>
 * <dt>b.1.2</dt>
 * <dd>{@code b=(y1-a)*(d+x1)^(-c)}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c= (log((y1-a)/b))/(log(d+x1))}</dd>
 * <dt>d.1.1</dt>
 * <dd>{@code d=(-(a-y1)/b)^(1/c)-x1}</dd>
 * </dl>
 */
public class ExpLinearModelOverLogX extends _ModelBase {

  /** the checker for parameter {@code a} */
  static final ParameterValueCheckerMinMax A = new ParameterValueCheckerMinMax(
      -1e100d, 1e100d);
  /** the checker for parameter {@code b} */
  static final ParameterValueCheckerMinMax B = new ParameterValueCheckerMinMax(
      1e-10d, 1e100d);
  /** the checker for parameter {@code c} */
  static final ParameterValueCheckerMinMax C = new ParameterValueCheckerMinMax(
      1e-100d, 5d);
  /** the checker for parameter {@code d} */
  static final ParameterValueCheckerMinMax D = new ParameterValueCheckerMinMax(
      -1e9d, 1e30d);

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
    final double a, res;

    res = ((a = parameters[0]) + (parameters[1] * _ModelBase
        ._exp_o_p(parameters[2], _ModelBase._log(parameters[3] + x))));
    return (MathUtils.isFinite(res) ? res//
        : (MathUtils.isFinite(a) ? a : 0d));
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double d, dx, c, dxc, b;

    gradient[0] = 1d;
    d = parameters[3];
    dx = d + x;
    c = parameters[2];
    dxc = _ModelBase._pow(dx, c);
    b = parameters[1];
    gradient[1] = _ModelBase._gradient(dxc, b);
    gradient[2] = _ModelBase._gradient(b * dxc * _ModelBase._log(dx), c);
    gradient[3] = _ModelBase._gradient(b * c * _ModelBase._pow(dx, c - 1d),
        d);
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
    out.append("*exp("); //$NON-NLS-1$
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
    final IMath closerB, closerD;
    final boolean negateB, negateD;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    if ((negatableRenderer != null) && (negatableRenderer.isNegative(1))) {
      negateB = true;
      closerB = out.sub();
    } else {
      negateB = false;
      closerB = out.add();
    }

    try {
      renderer.renderParameter(0, closerB);
      try (final IMath mul = closerB.mul()) {
        if (negateB) {
          negatableRenderer.renderNegatedParameter(1, mul);
        } else {
          renderer.renderParameter(1, mul);
        }

        try (final IMath exp = mul.exp()) {
          try (final IMath braces = exp.inBraces()) {
            try (final IMath mul2 = braces.mul()) {
              renderer.renderParameter(2, mul2);

              try (final IMath ln = mul2.ln()) {
                try (final IMath braces2 = ln.inBraces()) {

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
          }
        }
      }
    } finally {
      closerB.close();
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
  static final double _a_x0y0x1y1cd_1(final double x0, final double y0,
      final double x1, final double y1, final double c, final double d) {
    final double dx0c, dx1c;

    dx0c = _ModelBase._pow(d + x0, c);
    dx1c = _ModelBase._pow(d + x1, c);

    return ((y1 * dx0c) - (y0 * dx1c)) / (dx0c - dx1c);
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
  static final double _a_x0y0x1y1cd_2(final double x0, final double y0,
      final double x1, final double y1, final double c, final double d) {
    final double expclogx0d, expclogx1d;

    expclogx0d = _ModelBase._exp(c * _ModelBase._log(x0 + d));
    expclogx1d = _ModelBase._exp(c * _ModelBase._log(x1 + d));

    return ((y0 * expclogx1d) - (y1 * expclogx0d))
        / (expclogx1d - expclogx0d);
  }

  /**
   * compute {@code b} based on two points and {@code c} and {@code d}.
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
  static final double _b_x0y0x1y1cd_1(final double x0, final double y0,
      final double x1, final double y1, final double c, final double d) {
    return (y1 - y0) / (_ModelBase._exp(c * _ModelBase._log(x1 + d))
        - _ModelBase._exp(c * _ModelBase._log(x0 + d)));
  }

  /**
   * compute {@code b} based on two points and {@code c} and {@code d}.
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
  static final double _b_x0y0x1y1cd_2(final double x0, final double y0,
      final double x1, final double y1, final double c, final double d) {
    return (y1 - y0)
        / (_ModelBase._pow(d + x0, c) - _ModelBase._pow(d + x1, c));
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
    return (_ModelBase._log(y0 - a) - _ModelBase._log(y1 - a))
        / (_ModelBase._log(d + x0) - _ModelBase._log(d + x1));
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
  static final double _a_x0y0bcd_1(final double x0, final double y0,
      final double b, final double c, final double d) {
    return y0 - (b * _ModelBase._exp(c * _ModelBase._log(x0 + d)));
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
    return y0 - (b * _ModelBase._pow((d + x0), c));
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
  static final double _b_x0y0acd_1(final double x0, final double y0,
      final double a, final double c, final double d) {
    return _ModelBase._exp(-(c * _ModelBase._log(x0 + d))) * (y0 - a);
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
  static final double _b_x0y0acd_2(final double x0, final double y0,
      final double a, final double c, final double d) {
    return (y0 - a) * _ModelBase._pow((d + x0), (-c));
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
    return (_ModelBase._log((y0 - a) / b)) / (_ModelBase._log(d + x0));
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
    return _ModelBase._pow((-(a - y0) / b), (1d / c)) - x0;
  }

  /**
   * perform the fallback
   *
   * @param yMin
   *          the minimal y
   * @param yMax
   *          the maximak y
   * @param dest
   *          the destination to receive the guess
   * @param random
   *          the random number generator
   * @return {@code true}
   */
  static final boolean _fallback(final double yMin, final double yMax,
      final double[] dest, final Random random) {
    double maxY, minY;

    maxY = (yMax * Math.abs((1d + //
        Math.abs(0.05d * random.nextGaussian()))));
    minY = (yMin * Math.abs((1d - //
        Math.abs(0.05d * random.nextGaussian()))));

    maxY = Math.abs(maxY - minY);
    if (maxY < 1e-6d) {
      maxY = 1e-6d;
    }

    dest[0] = minY;
    dest[1] = maxY;

    do {
      dest[2] = -(random.nextInt(10) + random.nextGaussian());
    } while (dest[2] >= -1e-7d);

    dest[3] = (random.nextDouble() * 1e2d);
    return true;
  }

  /** the parameter guesser */
  private final class __ExpLinearModelOverLogXParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /**
     * Create the parameter guesser
     *
     * @param data
     *          the data
     */
    __ExpLinearModelOverLogXParameterGuesser(final IMatrix data) {
      super(data, 4, 2);
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExpLinearModelOverLogX.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected boolean fallback(final double[] points, final double[] dest,
        final Random random) {
      double maxY, minY;
      int i;

      findMaxY: {
        if (random.nextInt(10) <= 0) {
          maxY = this.m_maxY;
          if (MathUtils.isFinite(maxY)) {
            break findMaxY;
          }
        }
        maxY = Double.NEGATIVE_INFINITY;
        for (i = (points.length - 1); i > 0; i -= 2) {
          maxY = Math.max(maxY, points[i]);
        }
      }

      findMinY: {
        if (random.nextInt(10) <= 0) {
          minY = this.m_minY;
          if (MathUtils.isFinite(minY)) {
            break findMinY;
          }
        }
        minY = Double.POSITIVE_INFINITY;
        for (i = (points.length - 1); i > 0; i -= 2) {
          minY = Math.min(minY, points[i]);
        }
      }

      return ExpLinearModelOverLogX._fallback(minY, maxY, dest, random);
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      ExpLinearModelOverLogX._fallback(this.m_minY, this.m_maxY, dest,
          random);
    }

    /** {@inheritDoc} */
    @Override
    protected final void guessBasedOnPermutation(final double[] points,
        final double[] bestGuess, final double[] destGuess) {
      final double oldA, oldB, oldC, oldD;
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

            newA = ParameterValueChecker.choose(//
                ExpLinearModelOverLogX._a_x0y0x1y1cd_1(points[0],
                    points[1], points[2], points[3], (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX._a_x0y0x1y1cd_2(points[0],
                    points[1], points[2], points[3], (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX.A);
            if (ExpLinearModelOverLogX.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            newA = ParameterValueChecker.choose(//
                ExpLinearModelOverLogX._a_x0y0bcd_1(points[0], points[1],
                    (hasB ? newB : oldB), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX._a_x0y0bcd_1(points[0], points[1],
                    (hasB ? newB : oldB), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX.A);
            if (ExpLinearModelOverLogX.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }
          }
        }

        // find B
        findB: {
          if (!hasB) {
            newB = ParameterValueChecker.choose(//
                ExpLinearModelOverLogX._b_x0y0x1y1cd_1(points[0],
                    points[1], points[2], points[3], (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX._b_x0y0x1y1cd_2(points[0],
                    points[1], points[2], points[3], (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX.B);

            if (ExpLinearModelOverLogX.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = ParameterValueChecker.choose(//
                ExpLinearModelOverLogX._b_x0y0acd_1(points[0], points[1],
                    (hasA ? newA : oldA), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX._b_x0y0acd_1(points[0], points[1],
                    (hasA ? newA : oldA), (hasC ? newC : oldC),
                    (hasD ? newD : oldD)), //
                ExpLinearModelOverLogX.B);

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
            if (ExpLinearModelOverLogX.C.check(newC)) {
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
            if (ExpLinearModelOverLogX.D.check(newD)) {
              changed = hasD = true;
              break findC;
            }
          }
        }

        // OK, everything else has failed us
        emergency: {
          if (!(changed)) {

            if (!hasA) {
              newA = Math.nextUp(this.m_minY);
              if (ExpLinearModelOverLogX.A.check(newA)) {
                hasA = changed = true;
              }
              break emergency;
            }

            if (!hasB) {
              newB = (this.m_maxY - (hasA ? newA : this.m_minY));
              if (ExpLinearModelOverLogX.A.check(newB)) {
                hasB = changed = true;
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