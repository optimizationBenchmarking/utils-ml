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
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ParameterValueCheckerMinMaxAbs;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplePermutationBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model describing the relationship of input to output data similar to
 * an "exponential decay" according to the formula {@code a+(b*exp(c*x^d))}
 * .
 * </p>
 * <h2>Behavior Occurrences</h2>
 * <p>
 * We encounter this model in two basic forms when modeling optimization
 * processes, either with {@code b} and {@code d} both positive or both
 * negative. Below you can find two examples:
 * </p>
 * <ul>
 * <li>{@code 1.447E-5 +  2.017 ∗ exp(-1.642 ∗ x^ 0.265)}</li>
 * <li>{@code 0.231    + -0.231 ∗ exp(-43.66 ∗ x^-1.685)}</li>
 * </ul>
 * <h2>Derivatives</h2>
 * <p>
 * The derivatives have been obtained with http://www.numberempire.com/.
 * </p>
 * <ol>
 * <li>Original function: {@code a+(b*exp(c*x^d))}</li>
 * <li>{@code d/da}: {@code 1}</li>
 * <li>{@code d/db}: {@code exp(c*x^d)}</li>
 * <li>{@code d/dc}: {@code b*x^d*exp(c*x^d)}</li>
 * <li>{@code d/dd}: {@code b*c*x^d*exp(c*x^d)*log(x)}</li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1</dt>
 * <dd>{@code a=-(exp(c*x2^d)*y1-exp(c*x1^d)*y2)/(exp(c*x1^d)-exp(c*x2^d))}
 * <dt>a.2.2</dt>
 * <dd>
 * {@code a=(y2*exp(c*x1^d)-y1*(exp(c*x1^d))^(x2^d*x1^(-d)))/(exp(c*x1^d)-(exp(c*x1^d))^(x2^d*x1^(-d)))}
 * </dd>
 * <dt>b.2.1</dt>
 * <dd>{@code b=(y1-y2)/(exp(c*x1^d)-exp(c*x2^d))}</dd>
 * <dt>b.2.2</dt>
 * <dd>{@code b=(y1-y2)/(exp(c*x1^d)-(exp(c*x1^d))^(x2^d*x1^(-d)))}</dd>
 * </dl>
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=y-b*exp(c*x^d)}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=exp(-(c*x^d))*(y-a)}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c=log(y/b-a/b)/x^d}</dd>
 * <dt>d.1.1</dt>
 * <dd>{@code d=log(log(y/b-a/b)/c)/log(x)}</dd>
 * </dl>
 */
public final class ExponentialDecayModel extends _ModelBase {

  /** the checker for {@code a} */
  static final ParameterValueCheckerMinMax A = _ModelBase.CURVE_OFFSET;
  /** the checker for {@code b} */
  static final ParameterValueCheckerMinMaxAbs B = _ModelBase.CURVE_SPREAD;
  /** the checker for {@code c} */
  static final ParameterValueCheckerMinMaxAbs C = new ParameterValueCheckerMinMaxAbs(
      1e-10d, 1e5d);
  /** the checker for {@code d} */
  static final ParameterValueCheckerMinMaxAbs D = new ParameterValueCheckerMinMaxAbs(
      1e-5d, 50d);

  /** create the exponential decay model */
  public ExponentialDecayModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    final double a, b, c;
    double res;

    a = parameters[0];
    b = parameters[1];
    if (((c = parameters[2]) != 0d) && //
        (((res = _ModelBase._pow(x, parameters[3])) != 0d)) && //
        ((res *= c) != 0d)) {
      if (((res = _ModelBase._exp(res)) != 0d) && //
          ((res *= b) != 0d)) {
        if (((res += a) != 0d) && MathUtils.isFinite(res)) {
          return res;
        }
        return 0d;
      }
      return (MathUtils.isFinite(a) ? a : 0d);
    }
    res = (a + b);
    return (MathUtils.isFinite(res) ? res : 0d);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "generalized exponential decay"; //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double expcxd, xd, cxd, c, b, d;
    double g;

    gradient[0] = 1d;
    d = parameters[3];
    xd = _ModelBase._pow(x, d);
    c = parameters[2];
    if ((xd == 0d) || (c == 0d)) {
      cxd = 0d;
    } else {
      cxd = c * xd;
    }
    expcxd = _ModelBase._exp(cxd);

    b = parameters[1];
    gradient[1] = _ModelBase._gradient(expcxd, b);

    if (b != 0d) {
      g = b * xd * expcxd;
      gradient[2] = _ModelBase._gradient(g, c);
      gradient[3] = (((g *= c) != 0d)
          ? _ModelBase._gradient(g * _ModelBase._log(x), d) : 0d);
      return;
    }
    gradient[2] = gradient[3] = 0d;
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
    out.append('*');
    x.mathRender(out, renderer);
    out.append('^');
    renderer.renderParameter(3, out);
    out.append(')');
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "null", "resource" })
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    final INegatableParameterRenderer negatableRenderer;
    final IMath closer;
    final boolean negate;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    if ((negatableRenderer != null) && (negatableRenderer.isNegative(1))) {
      negate = true;
      closer = out.sub();
    } else {
      negate = false;
      closer = out.add();
    }

    try {
      renderer.renderParameter(0, closer);

      try (final IMath mul = closer.mul()) {
        if (negate) {
          negatableRenderer.renderNegatedParameter(1, mul);
        } else {
          renderer.renderParameter(1, mul);
        }
        try (final IMath exp = mul.exp()) {
          try (final IMath braces = exp.inBraces()) {
            try (final IMath mul2 = braces.mul()) {
              renderer.renderParameter(2, mul2);
              try (final IMath pow = mul2.pow()) {
                x.mathRender(pow, renderer);
                renderer.renderParameter(3, pow);
              }
            }
          }
        }
      }

    } finally {
      closer.close();
    }
  }

  /** {@inheritDoc} */
  @Override
  public IParameterGuesser createParameterGuesser(final IMatrix data) {
    return new __DecayModelParameterGuesser(data);
  }

  /**
   * compute {@code a} based on two points and {@code c} and {@code d}.
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
   * @param d
   *          the {@code d} value
   * @return the guess for {@code a}
   */
  static final double _a_x1y1x2y2cd(final double x1, final double y1,
      final double x2, final double y2, final double c, final double d) {
    final double x1d, x2d, ecx1d, ecx2d, pp;

    x1d = _ModelBase._pow(x1, d);
    ecx1d = _ModelBase._exp(c * x1d);
    x2d = _ModelBase._pow(x2, d);
    ecx2d = _ModelBase._exp(c * x2d);
    pp = _ModelBase._pow((ecx1d), (x2d / x1d));

    return ParameterValueChecker.choose(//
        -((ecx2d * y1) - (ecx1d * y2)) / (ecx1d - ecx2d), //
        ((y2 * ecx1d) - (y1 * pp)) / (ecx1d - pp), //
        ExponentialDecayModel.A);
  }

  /**
   * compute {@code b} based on two points and {@code c} and {@code d}.
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
   * @param d
   *          the {@code d} value
   * @return the guess for {@code b}
   */
  static final double _b_x1y1x2y2cd(final double x1, final double y1,
      final double x2, final double y2, final double c, final double d) {
    final double x1d, x2d, ecx1d, ecx2d;

    x1d = _ModelBase._pow(x1, d);
    x2d = _ModelBase._pow(x2, d);
    ecx1d = _ModelBase._exp(c * x1d);
    ecx2d = _ModelBase._exp(c * x2d);

    return ParameterValueChecker.choose(//
        (y1 - y2) / (ecx1d - ecx2d), //
        (y1 - y2) / (ecx1d - _ModelBase._pow((ecx1d), (x2d / x1d))), //
        ExponentialDecayModel.B);
  }

  /**
   * Compute {@code a} from one point and {@code b}, {@code c}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code a}
   */
  static final double _a_x1y1bcd(final double x1, final double y1,
      final double b, final double c, final double d) {
    return y1 - (b * _ModelBase._exp(c * _ModelBase._pow(x1, d)));
  }

  /**
   * Compute {@code b} from one point and {@code a}, {@code c}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code b}
   */
  static final double _b_x1y1acd(final double x1, final double y1,
      final double a, final double c, final double d) {
    return _ModelBase._exp(-(c * _ModelBase._pow(x1, d))) * (y1 - a);
  }

  /**
   * Compute {@code c} from one point and {@code a}, {@code b}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code c}
   */
  static final double _c_x1y1abd(final double x1, final double y1,
      final double a, final double b, final double d) {
    final double l;

    l = _ModelBase._log((y1 / b) - (a / b));
    return ParameterValueChecker.choose(//
        l / _ModelBase._pow(x1, d), //
        l * _ModelBase._pow(x1, -d), ExponentialDecayModel.C);
  }

  /**
   * Compute {@code d} from one point and {@code a}, {@code b}, and
   * {@code c}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @return the value of {@code d}
   */
  static final double _d_x1y1abc(final double x1, final double y1,
      final double a, final double b, final double c) {
    return _ModelBase._log(_ModelBase._log((y1 / b) - (a / b)) / c)
        / _ModelBase._log(x1);
  }

  /**
   * the internal fallback routine
   *
   * @param minY
   *          the minimal y coordinate
   * @param maxY
   *          the maximum y coordinate
   * @param dest
   *          the destination array
   * @param random
   *          the random number generator
   */
  static final void _fallback(final double minY, final double maxY,
      final double[] dest, final Random random) {
    if (random.nextBoolean()) {
      dest[0] = minY;
      dest[1] = (maxY - minY);
      dest[2] = -random.nextDouble();
      dest[3] = random.nextInt(5) + random.nextDouble();
    } else {
      dest[0] = maxY;
      dest[1] = (minY - maxY);
      dest[2] = -(random.nextDouble() + random.nextInt(200));
      dest[3] = -(random.nextInt(5) + random.nextDouble());
    }
  }

  /** the parameter guesser */
  private class __DecayModelParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __DecayModelParameterGuesser(final IMatrix data) {
      super(data, ExponentialDecayModel.this.getParameterCount(),
          ExponentialDecayModel.this.getParameterCount() - 1);
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExponentialDecayModel.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean fallback(final double[] points,
        final double[] dest, final Random random) {
      final double[] minMax;
      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);
      ExponentialDecayModel._fallback(minMax[0], minMax[1], dest, random);
      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      final double[] minMax;
      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY, null,
          random);
      ExponentialDecayModel._fallback(minMax[0], minMax[1], dest, random);
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

            newA = ExponentialDecayModel._a_x1y1x2y2cd(points[0],
                points[1], points[2], points[3], (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (ExponentialDecayModel.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            newA = ExponentialDecayModel._a_x1y1bcd(points[0], points[1],
                (hasB ? newB : oldB), (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (ExponentialDecayModel.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }
          }
        }

        // find B
        findB: {
          if (!hasB) {
            newB = ExponentialDecayModel._b_x1y1x2y2cd(points[0],
                points[1], points[2], points[3], (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (ExponentialDecayModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = ExponentialDecayModel._b_x1y1acd(points[0], points[1],
                (hasA ? newA : oldA), (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (ExponentialDecayModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }
          }
        }

        // find C
        findC: {
          if (!hasC) {
            newC = ExponentialDecayModel._c_x1y1abd(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasD ? newD : oldD));
            if (ExponentialDecayModel.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }
          }
        }

        // find D
        findD: {
          if (!hasD) {
            newD = ExponentialDecayModel._d_x1y1abc(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasC ? newC : oldC));
            if (ExponentialDecayModel.D.check(newD)) {
              changed = hasD = true;
              break findD;
            }
          }
        }

        // OK, everything else has failed us
        emergency: {
          if (!(changed)) {

            if (!(hasA)) {
              newA = Math.min(points[1], Math.min(points[3], points[5]));
              if (ExponentialDecayModel.A.check(newA)) {
                hasA = changed = true;
                break emergency;
              }
            }

            if (!(hasB)) {
              newB = (Math.max(points[1], Math.max(points[3], points[5])) - //
                  (hasA ? newA
                      : Math.min(points[1],
                          Math.min(points[3], points[5]))));
              if (ExponentialDecayModel.B.check(newB)) {
                hasB = changed = true;
                break emergency;
              }
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
