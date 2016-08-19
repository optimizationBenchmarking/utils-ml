package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueChecker;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueCheckerMinMaxAbs;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplePermutationBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model function which may be suitable to model how time-objective value
 * relationships of optimization processes behave. Qi Qi has discovered
 * this model: {@code a/(1+b*x^c)}.
 * </p>
 * <p>
 * It is somewhat similar to a logistic model and over a log-scaled
 * {@code x}-axis: Originally, we fitted {@code a/(1+exp(b+c*log(x)))},
 * which corresponds to {@code a/(1+exp(b)*(x^c))}, which, in turn, can be
 * simplified to {@code a/(1+b*x^c)} in the fitting procedure, since
 * {@code exp(b)} is constant with respect to {@code x} and then we can try
 * to find its value directly as well.
 * </p>
 * <p>
 * This model is also somewhat similar to my previous exponential decay
 * model {@code -(exp(a*(x^b))-1)} but seems to fit better and does not
 * require data normalization.
 * </p>
 * <h2>Derivatives</h2>
 * <p>
 * The derivatives have been obtained with http://www.numberempire.com/.
 * </p>
 * <ol>
 * <li>Original function: {@code a/(1+b*x^c)}</li>
 * <li>{@code d/da}: {@code 1/(1+b*x^c)}</li>
 * <li>{@code d/db}: {@code -(a*x^c)/         (1 + 2*b*x^c + b^2*x^(2*c))}
 * </li>
 * <li>{@code d/dc}: {@code -(a*b*x^c*log(x))/(1 + 2*b*x^c + b^2*x^(2*c))}
 * </li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>One Known Point</h3>
 * <ol>
 * <li>{@code a}: {@code a=(b*x^c+1)*y}</li>
 * <li>{@code b}: {@code b=-(y-a)/(x^c*y)}</li>
 * <li>{@code c}: {@code c=log(a/(b*y)-1/b)/log(x)},
 * {@code c = (log((a-y)/(b*y)))/(log(x))}</li>
 * </ol>
 * <h3>Two Known Points</h3>
 * <ol>
 * <li>{@code a}: {@code a=((x2^c-x1^c)*y1*y2)/(x2^c*y2-x1^c*y1)}</li>
 * <li>{@code b}: {@code b=-(y2-y1)/(x2^c*y2-x1^c*y1)}</li>
 * <li>{@code b}:
 * {@code b=exp((log(x1)*log(a-y2)-log(x2)*log(a-y1)-log(x1)*log(y2)+log(x2)*log(y1))/(log(x1)-log(x2)))}
 * </li>
 * </ol>
 */
public final class LogisticModelOverLogX extends _ModelBase {

  /** the checker for parameter {@code a} */
  static final ParameterValueCheckerMinMaxAbs A = new ParameterValueCheckerMinMaxAbs(
      1e-12d, 1e100d);
  /** the checker for parameter {@code b} */
  static final ParameterValueCheckerMinMaxAbs B = new ParameterValueCheckerMinMaxAbs(
      1e-10d, 1e10d);
  /** the checker for parameter {@code c} */
  static final ParameterValueCheckerMinMaxAbs C = new ParameterValueCheckerMinMaxAbs(
      1e-4d, 1e4d);

  /** create */
  public LogisticModelOverLogX() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "logistic model"; //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public double value(final double x, final double[] parameters) {
    final double a;
    double res, b;

    if ((a = parameters[0]) != 0d) {
      if (((b = parameters[1]) != 0d) && //
          ((res = _ModelBase._pow(x, parameters[2])) != 0d)) {
        if (((res = (1d + (b * res))) != 0d) && //
            MathUtils.isFinite(res) && //
            MathUtils.isFinite(res = (a / res))) {
          return res;
        }
        return 0d;
      }
      return (MathUtils.isFinite(a) ? a : 0d);
    }
    return 0d;
  }

  /** {@inheritDoc} */
  @Override
  public void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double a, b, c, xc, bxc, axc, div;

    c = parameters[2];
    xc = _ModelBase._pow(x, c);

    if (Math.abs(xc) <= 0d) {
      gradient[0] = 1d;
      gradient[1] = gradient[2] = 0d;
      return;
    }

    b = parameters[1];
    bxc = (b * xc);
    if (bxc <= 0) {
      gradient[0] = 1d;
      gradient[1] = gradient[2] = 0d;
      return;
    }

    a = parameters[0];
    gradient[0] = _ModelBase._gradient((1d / (1d + bxc)), a);

    axc = (a * xc);
    if (axc == 0d) {
      gradient[1] = gradient[2] = 0d;
      return;
    }

    div = _ModelBase._add(1d, 2d * bxc, bxc * bxc);
    gradient[1] = _ModelBase._gradient(((-axc) / div), b);
    gradient[2] = _ModelBase
        ._gradient(((-(b * axc * _ModelBase._log(x))) / div), c);
  }

  /** {@inheritDoc} */
  @Override
  public int getParameterCount() {
    return 3;
  }

  /** {@inheritDoc} */
  @Override
  public void mathRender(final ITextOutput out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    renderer.renderParameter(0, out);
    out.append('/');
    out.append('(');
    out.append('1');
    out.append('+');
    renderer.renderParameter(1, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('^');
    renderer.renderParameter(2, out);
    out.append(')');
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "null", "resource" })
  @Override
  public void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    final INegatableParameterRenderer negatableRenderer;
    final IMath closer;
    final boolean negate;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    try (final IMath div = out.div()) {
      renderer.renderParameter(0, div);

      if ((negatableRenderer != null)
          && (negatableRenderer.isNegative(1))) {
        negate = true;
        closer = div.sub();
      } else {
        negate = false;
        closer = div.add();
      }

      try {
        try (final IText num = closer.number()) {
          num.append('1');
        }
        try (final IMath mul = closer.mul()) {

          if (negate) {
            negatableRenderer.renderNegatedParameter(1, mul);
          } else {
            renderer.renderParameter(1, mul);
          }

          try (final IMath pow = mul.pow()) {
            x.mathRender(pow, renderer);
            renderer.renderParameter(2, pow);
          }
        }

      } finally {
        closer.close();
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public IParameterGuesser createParameterGuesser(final IMatrix data) {
    return new _LogisticModelOverLogXParameterGuesser(data);
  }

  /**
   * Compute {@code a} from one point {@code (x,y)} and known {@code b} and
   * {@code c} values.
   *
   * @param x
   *          the {@code x}-coordinate of the point
   * @param y
   *          the {@code y}-coordinate of the point
   * @param b
   *          the {@code b} value
   * @param c
   *          the {@code c} value
   * @return the {@code a} value
   */
  static final double _a_xybc(final double x, final double y,
      final double b, final double c) {
    return ((x <= 0d) ? y : (((b * _ModelBase._pow(x, c)) + 1d) * y));
  }

  /**
   * Compute {@code a} from two points {@code (x1,y1)} and {@code (x2,y2)}
   * and a known {@code c} value.
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param x2
   *          the {@code x}-coordinate of the second point
   * @param y2
   *          the {@code y}-coordinate of the second point
   * @param c
   *          the {@code c} value
   * @return the {@code a} value
   */
  static final double _a_x1y1x2y2c(final double x1, final double y1,
      final double x2, final double y2, final double c) {
    final double x2c, x1c;

    x1c = _ModelBase._pow(x1, c);
    x2c = _ModelBase._pow(x2, c);
    return (((x2c - x1c) * y1 * y2) / ((x2c * y2) - (x1c * y1)));
  }

  /**
   * Compute {@code b} from two points {@code (x1,y1)} and {@code (x2,y2)}
   * and a known {@code c} value.
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param x2
   *          the {@code x}-coordinate of the second point
   * @param y2
   *          the {@code y}-coordinate of the second point
   * @param c
   *          the {@code c} value
   * @return the {@code b} value
   */
  static final double _b_x1y1x2y2c(final double x1, final double y1,
      final double x2, final double y2, final double c) {
    return ((y1 - y2)
        / ((_ModelBase._pow(x2, c) * y2) - (_ModelBase._pow(x1, c) * y1)));
  }

  /**
   * Compute {@code b} from two points {@code (x1,y1)} and {@code (x2,y2)}
   * and a known {@code a} value.
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param x2
   *          the {@code x}-coordinate of the second point
   * @param y2
   *          the {@code y}-coordinate of the second point
   * @param a
   *          the {@code a} value
   * @return the {@code b} value
   */
  static final double _b_x1y1x2y2a(final double x1, final double y1,
      final double x2, final double y2, final double a) {
    final double lx1, lx2;

    lx1 = _ModelBase._log(x1);
    lx2 = _ModelBase._log(x2);
    return _ModelBase._exp(_ModelBase._add(//
        (lx1 * _ModelBase._log(a - y2)), //
        -(lx2 * _ModelBase._log(a - y1)), //
        -(lx1 * _ModelBase._log(y2)), //
        (lx2 * _ModelBase._log(y1))) / (lx1 - lx2));
  }

  /**
   * Compute {@code b} from one point {@code (x,y)} and known {@code a} and
   * {@code c} values.
   *
   * @param x
   *          the {@code x}-coordinate of the point
   * @param y
   *          the {@code y}-coordinate of the point
   * @param a
   *          the {@code a} value
   * @param c
   *          the {@code c} value
   * @return the {@code b} value
   */
  static final double _b_xyac(final double x, final double y,
      final double a, final double c) {
    final double xc, xcy;

    xc = _ModelBase._pow(x, c);
    xcy = (xc * y);

    return ParameterValueChecker.choose(//
        ((a - y) / xcy), //
        ((a / xcy) - (1d / xc)), //
        LogisticModelOverLogX.B);
  }

  /**
   * Compute {@code c} from one point {@code (x,y)} and known {@code a} and
   * {@code b} values.
   *
   * @param x
   *          the {@code x}-coordinate of the point
   * @param y
   *          the {@code y}-coordinate of the point
   * @param a
   *          the {@code a} value
   * @param b
   *          the {@code b} value
   * @return the {@code c} value
   */
  static final double _c_xyab(final double x, final double y,
      final double a, final double b) {
    final double lx, by;

    lx = _ModelBase._log(x);
    if ((b <= (-1d)) && (0d < x) && (x < 1d) && (Math.abs(a) <= 0d)
        && (Math.abs(y) <= 0d)) {
      return Math.nextUp(Math.nextUp(//
          _ModelBase._log(-1d / b) / lx));
    }

    by = b * y;
    return ParameterValueChecker.choose(//
        _ModelBase._log((a / by) - (1 / b)) / lx, //
        _ModelBase._log((a - y) / by) / lx, //
        LogisticModelOverLogX.C);
  }

  /**
   * the internal fallback routine
   *
   * @param signChoice
   *          {@code true} if all parameters are positive, {@code false} if
   *          {@code b} and {@code d} should be negative
   * @param spread
   *          the parameter spread
   * @param dest
   *          the destination array
   * @param offset
   *          the parameter offset
   * @param random
   *          the random number generator
   */
  static final void _fallback(final boolean signChoice,
      final double spread, final double[] dest, final int offset,
      final Random random) {
    if (signChoice) {
      dest[offset] = _ModelBase
          ._exp((_ModelBase._log(spread) + Math.abs(random.nextGaussian()))
              - (6d * Math.abs(random.nextGaussian())));
      dest[offset + 1] = _ModelBase
          ._exp((random.nextInt(24) - 18) + random.nextDouble());
      dest[offset + 2] = 15d * random.nextDouble();
    } else {
      dest[offset] = -spread * (2d
          * _ModelBase._exp(-(random.nextDouble() + random.nextInt(10))));
      dest[offset + 1] = 100d
          * _ModelBase._exp(-(random.nextDouble() + random.nextInt(10)));
      dest[offset + 2] = 5d * random.nextDouble();
    }
  }

  /** the parameter guesser */
  private final class _LogisticModelOverLogXParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /**
     * Create the parameter guesser
     *
     * @param data
     *          the data
     */
    _LogisticModelOverLogXParameterGuesser(final IMatrix data) {
      super(data, LogisticModelOverLogX.this.getParameterCount(), //
          (LogisticModelOverLogX.this.getParameterCount() - 1));
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean fallback(final double[] points,
        final double[] dest, final Random random) {
      final double[] minMax;
      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);
      LogisticModelOverLogX._fallback(random.nextBoolean(),
          minMax[1] - minMax[0], dest, 0, random);
      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      final double[] minMax;
      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY, null,
          random);
      LogisticModelOverLogX._fallback(random.nextBoolean(),
          minMax[1] - minMax[0], dest, 0, random);
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return LogisticModelOverLogX.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected final void guessBasedOnPermutation(final double[] points,
        final double[] bestGuess, final double[] destGuess) {
      final double x0, y0, x1, y1/* , x2, y2 */, oldA, oldB, oldC;
      double newA, newB, newC;
      boolean hasA, hasB, hasC, changed;

      hasA = hasB = hasC = false;
      x0 = points[0];
      y0 = points[1];
      x1 = points[2];
      y1 = points[3];
      // x2 = points[4];
      // y2 = points[5];
      newA = oldA = bestGuess[0];
      newB = oldB = bestGuess[1];
      newC = oldC = bestGuess[2];

      changed = true;
      while (changed) {
        changed = false;

        if (!hasB) {
          // find B based on the existing or new A and C values
          newB = LogisticModelOverLogX._b_x1y1x2y2a(x0, y0, x1, y1,
              (hasA ? newA : oldA));
          if (LogisticModelOverLogX.B.check(newB)) {
            changed = hasB = true;
          } else {
            newB = LogisticModelOverLogX._b_xyac(x0, y0,
                (hasA ? newA : bestGuess[0]), (hasC ? newC : oldC));
            if (LogisticModelOverLogX.B.check(newB)) {
              changed = hasB = true;
            } else {
              newB = LogisticModelOverLogX._b_x1y1x2y2c(x0, y0, x1, y1,
                  (hasC ? newC : oldC));
              if (LogisticModelOverLogX.B.check(newB)) {
                changed = hasB = true;
              }
            }
          }
        }

        if (!hasC) {
          // find C based on the existing or new A and B values
          newC = LogisticModelOverLogX._c_xyab(x0, y0,
              (hasA ? newA : oldA), (hasB ? newB : oldB));
          if (LogisticModelOverLogX.C.check(newC)) {
            changed = hasC = true;
          }
        }

        if (!hasA) {
          findA: {
            // find A based on the existing or new B and C values
            if (hasB) {
              newA = LogisticModelOverLogX._a_xybc(x0, y0, newB,
                  (hasC ? newC : oldC));
              if (LogisticModelOverLogX.A.check(newA)) {
                changed = hasA = true;
                break findA;
              }
            }

            newA = LogisticModelOverLogX._a_x1y1x2y2c(x0, y0, x1, y1,
                (hasC ? newC : oldC));
            if (LogisticModelOverLogX.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            if (!hasB) {
              newA = LogisticModelOverLogX._a_xybc(x0, y0, oldB,
                  (hasC ? newC : oldC));
              if (LogisticModelOverLogX.A.check(newA)) {
                changed = hasA = true;
                break findA;
              }
            }

            if (!changed) {
              if (Math.abs(x0) <= 0d) {
                newA = y0;
                changed = hasA = true;
                break findA;
              }
              if (Math.abs(x1) <= 0d) {
                newA = y1;
                changed = hasA = true;
                break findA;
              }
              // if (Math.abs(x2) <= 0d) {
              // newA = y2;
              // changed = hasA = true;
              // break findA;
              // }
            }
          }
        }
      }

      destGuess[0] = newA;
      destGuess[1] = newB;
      destGuess[2] = newC;
    }
  }
}