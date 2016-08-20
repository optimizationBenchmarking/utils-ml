package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplingBasedParameterGuesser;
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
    double res;

    l = _ModelBase._log((y1 / b) - (a / b));
    res = l / _ModelBase._pow(x1, d);
    if (MathUtils.isFinite(res)) {
      return res;
    }
    return l * _ModelBase._pow(x1, -d);
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
    double temp;
    int steps;

    if (random.nextBoolean()) {
      dest[0] = minY;
      dest[1] = (maxY - minY);
      steps = 100;
      do {
        dest[2] = temp = -_ModelBase._exp(-7d * random.nextDouble());
      } while ((((--steps) > 0) && (temp >= -1e-13d))
          || (!(MathUtils.isFinite(temp))));
      steps = 100;
      do {
        dest[3] = temp = 5d * _ModelBase._exp(-5d * random.nextDouble());
      } while ((((--steps) > 0) && (temp <= 1e-13d))
          || (!(MathUtils.isFinite(temp))));
    } else {
      dest[0] = maxY;
      dest[1] = (minY - maxY);
      steps = 100;
      do {
        dest[2] = temp = -(random.nextDouble() + random.nextInt(200));
      } while ((((--steps) > 0) && (temp >= -1e-10d))
          || (!(MathUtils.isFinite(temp))));
      steps = 100;
      do {
        dest[3] = temp = -(1d / (Math.abs(random.nextGaussian())
            + (1e-7d * random.nextDouble())));
      } while ((((--steps) > 0) && (temp >= -1e-11d))
          || (!(MathUtils.isFinite(temp))));
    }
  }

  /** the parameter guesser */
  private class __DecayModelParameterGuesser
      extends SamplingBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __DecayModelParameterGuesser(final IMatrix data) {
      super(data, 3, 4);
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
      double temp;

      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);
      ExponentialDecayModel._fallback(minMax[0], minMax[1], dest, random);

      switch (random.nextInt(3)) {
        case 0: {
          temp = ExponentialDecayModel._c_x1y1abd(points[0], points[1],
              dest[0], dest[1], dest[3]);
          if (_ModelBase._check(temp, dest[2], 1e-13d)) {
            dest[2] = temp;
          }
          temp = ExponentialDecayModel._d_x1y1abc(points[0], points[1],
              dest[0], dest[1], dest[2]);
          if (_ModelBase._check(temp, dest[3], 1e-13d)) {
            dest[3] = temp;
          }
          break;
        }
        default: {
          temp = ExponentialDecayModel._d_x1y1abc(points[0], points[1],
              dest[0], dest[1], dest[2]);
          if (_ModelBase._check(temp, dest[3], 1e-13d)) {
            dest[3] = temp;
          }
          temp = ExponentialDecayModel._c_x1y1abd(points[0], points[1],
              dest[0], dest[1], dest[3]);
          if (_ModelBase._check(temp, dest[2], 1e-13d)) {
            dest[2] = temp;
          }
        }
      }

      switch (random.nextInt(3)) {
        case 0: {
          temp = ExponentialDecayModel._a_x1y1bcd(points[0], points[1],
              dest[1], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[0], 1e-13d)) {
            dest[0] = temp;
          }

          temp = ExponentialDecayModel._b_x1y1acd(points[0], points[1],
              dest[0], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[1], 1e-13d)) {
            dest[1] = temp;
          }
          break;
        }
        default: {
          temp = ExponentialDecayModel._b_x1y1acd(points[0], points[1],
              dest[0], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[1], 1e-13d)) {
            dest[1] = temp;
          }
          temp = ExponentialDecayModel._a_x1y1bcd(points[0], points[1],
              dest[1], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[0], 1e-13d)) {
            dest[0] = temp;
          }

          break;
        }
      }

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
  }
}
