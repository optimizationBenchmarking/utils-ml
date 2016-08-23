package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.ImprovingSamplingBasedParameterGuesser;
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
      if (((res = Math.exp(res)) != 0d) && //
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
    expcxd = Math.exp(cxd);

    b = parameters[1];
    gradient[1] = _ModelBase._gradient(expcxd, b);

    if (b != 0d) {
      g = b * xd * expcxd;
      gradient[2] = _ModelBase._gradient(g, c);
      gradient[3] = (((g *= c) != 0d)
          ? _ModelBase._gradient(g * Math.log(x), d) : 0d);
    } else {
      gradient[2] = gradient[3] = 0d;
    }
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

  /** the parameter guesser */
  private class __DecayModelParameterGuesser
      extends ImprovingSamplingBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __DecayModelParameterGuesser(final IMatrix data) {
      super(data, 2, 4, new int[] { 1, 1, 2, 1, });
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExponentialDecayModel.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("incomplete-switch")
    @Override
    protected final double improveParameter(final int variant,
        final int parameter, final int guesser, final double[] points,
        final double[] parameters, final Random random) {
      final double x0, y0, a, b, c, d;

      x0 = points[0];
      y0 = points[1];
      a = parameters[0];
      b = parameters[1];
      c = parameters[2];
      d = parameters[3];

      switch (parameter) {
        case 0: {// a
          return y0 - (b * Math.exp(c * _ModelBase._pow(x0, d)));
        }

        case 1: {// b
          return Math.exp(-(c * _ModelBase._pow(x0, d))) * (y0 - a);
        }

        case 2: {// c
          switch (guesser) {
            case 0: {
              return ((Math.log((y0 / b) - (a / b)))
                  / _ModelBase._pow(x0, d));
            }

            case 1: {
              return ((Math.log((y0 / b) - (a / b)))
                  * _ModelBase._pow(x0, -d));
            }
          }
          break;
        }

        case 3: {// d
          return Math.log(Math.log((y0 / b) - (a / b)) / c) / Math.log(x0);
        }
      }

      return super.improveParameter(variant, parameter, guesser, points,
          parameters, random);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("incomplete-switch")
    @Override
    protected final boolean checkParameter(final int variant,
        final int parameter, final double newValue,
        final double[] parameters) {

      if (variant == 0) {
        switch (parameter) {
          case 1: {
            return ((newValue > 1e-13d) && (newValue < 1e100d));
          }
          case 2: {
            return ((newValue > -1e4d) && (newValue < -1e-13d));
          }
          case 3: {
            return ((newValue > 1e-13d) && (newValue < 1e4d));
          }
        }
      } else {
        switch (parameter) {
          case 1: {
            return ((newValue < -1e-13d) && (newValue > -1e100d));
          }
          case 2: {
            return ((newValue > -1e5d) && (newValue < -1e-13d));
          }
          case 3: {
            return ((newValue < -1e-13d) && (newValue > -1e4d));
          }
        }
      }

      return ((newValue > -1e100d) && (newValue < 1e100d));
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean guess(final int variant, final double[] points,
        final double[] dest, final Random random) {
      final double[] minMax;
      double temp;
      int steps;

      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);

      if (variant == 0) {
        dest[0] = minMax[0];
        dest[1] = (minMax[1] - minMax[0]);
        steps = 100;

        do {
          temp = -Math.exp(-7d * random.nextDouble());
        } while (((--steps) > 0)
            && (!(this.checkParameter(0, 2, temp, dest))));
        dest[2] = temp;

        steps = 100;
        do {
          temp = 5d * Math.exp(-5d * random.nextDouble());
        } while (((--steps) > 0)
            && (!(this.checkParameter(0, 3, temp, dest))));
        dest[3] = temp;

      } else {
        dest[0] = minMax[1];
        dest[1] = (minMax[0] - minMax[1]);

        steps = 100;
        do {
          temp = -(random.nextDouble() + random.nextInt(200));
        } while (((--steps) > 0)
            && (!(this.checkParameter(1, 2, temp, dest))));
        dest[2] = temp;

        steps = 100;
        do {
          temp = -(1d / (Math.abs(random.nextGaussian())
              + (1e-7d * random.nextDouble())));
        } while (((--steps) > 0)
            && (!(this.checkParameter(1, 3, temp, dest))));
        dest[3] = temp;
      }

      return true;
    }

  }
}
