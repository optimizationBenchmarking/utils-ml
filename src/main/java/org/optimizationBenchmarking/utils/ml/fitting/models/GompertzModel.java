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
 * an Gompertz function (https://en.wikipedia.org/wiki/Gompertz_function)
 * according to the formula {@code a+(b*exp(c*exp(d*x)))}.
 * </p>
 * <h2>Derivatives</h2>
 * <p>
 * The derivatives have been obtained with http://www.numberempire.com/.
 * </p>
 * <ol>
 * <li>Original function: {@code a+(b*exp(c*exp(d*x)))}</li>
 * <li>{@code d/da}: {@code 1}</li>
 * <li>{@code d/db}: {@code exp(c*exp(d*x))}</li>
 * <li>{@code d/dc}: {@code b*exp(c*exp(d*x)+d*x)}</li>
 * <li>{@code d/dd}: {@code b*c*x*exp(c*exp(d*x)+d*x)}</li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1</dt>
 * <dd>
 * {@code a=-(exp(c*exp(d*x2))*y1-exp(c*exp(d*x1))*y2)/(exp(c*exp(d*x1))-exp(c*exp(d*x2)))}
 * </dd>
 * <dt>b.2.1</dt>
 * <dd>{@code b=(y1-y2)/(exp(c*exp(d*x1))-exp(c*exp(d*x2)))}</dd>
 * </dl>
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=y1-b*exp(c*exp(d*x1))}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=exp(-(c*exp(d*x1)))*(y1-a)}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c=exp(-(d*x1))*log(y1/b-a/b)}</dd>
 * <dt>d.1.1</dt>
 * <dd>{@code d=log(log(y1/b-a/b)/c)/x1}</dd>
 * </dl>
 */
public final class GompertzModel extends _ModelBase {
  /** create the exponential decay model */
  public GompertzModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    final double a, res;

    res = ((_ModelBase._exp_o_p(_ModelBase._exp_o_p(parameters[3], x),
        parameters[2]) * parameters[1]) + (a = parameters[0]));
    return (MathUtils.isFinite(res) ? res//
        : (MathUtils.isFinite(a) ? a : 0d));
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "Gompertz function"; //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final boolean xIsZero;
    final double b, c, d;
    double expdx, cexpdx, dx;

    gradient[0] = 1d;
    xIsZero = (x == 0d);

    d = parameters[3];
    c = parameters[2];
    if ((d == 0d) || xIsZero || ((dx = (d * x)) != dx)) {
      dx = 0d;
      expdx = 1d;
      cexpdx = (c == c) ? c : 0d;
    } else {
      expdx = Math.exp(dx);
      if (expdx != expdx) {
        expdx = 0d;
        cexpdx = 0d;
      } else {
        cexpdx = (c * expdx);
        if (cexpdx != cexpdx) {
          cexpdx = 0d;
        }
      }
    }

    b = parameters[1];
    gradient[1] = _ModelBase._gradient(Math.exp(cexpdx), b);
    if ((b != 0d) && (b == b)) {
      cexpdx = (b * Math.exp(cexpdx + dx));
      gradient[2] = _ModelBase._gradient(cexpdx, c);
      gradient[3] = _ModelBase._gradient((cexpdx * c * x), d);
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
  @Override // a+(b*exp(c*exp(d*x)))
  public final void mathRender(final ITextOutput out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    renderer.renderParameter(0, out);
    out.append('+');
    renderer.renderParameter(1, out);
    out.append("*exp("); //$NON-NLS-1$
    renderer.renderParameter(2, out);
    out.append("*exp("); //$NON-NLS-1$
    renderer.renderParameter(3, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append(')');
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
              try (final IMath exp2 = mul2.exp()) {
                try (final IMath braces2 = exp2.inBraces()) {
                  try (final IMath mul3 = braces2.mul()) {
                    renderer.renderParameter(3, mul3);
                    x.mathRender(mul3, renderer);
                  }
                }
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
    return new __GompertzModelParameterGuesser(data);
  }

  /** the parameter guesser */
  private class __GompertzModelParameterGuesser
      extends ImprovingSamplingBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __GompertzModelParameterGuesser(final IMatrix data) {
      super(data, 2, 4, new int[] { 2, 2, 2, 2, });
    }

    /** {@inheritDoc} */
    @SuppressWarnings("incomplete-switch")
    @Override
    protected final double improveParameter(final int variant,
        final int parameter, final int guesser, final double[] points,
        final double[] parameters, final Random random) {
      final double x0, y0, a, b, c, d, expcexpdx1, expcexpdx2;

      x0 = points[0];
      y0 = points[1];
      a = parameters[0];
      b = parameters[1];
      c = parameters[2];
      d = parameters[3];

      switch (parameter) {
        case 0: {// a
          switch (guesser) {
            case 0: {
              return (y0 - (b
                  * _ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x0))));
            }
            case 1: {
              if (points.length > 2) {
                expcexpdx1 = _ModelBase._exp_o_p(c,
                    _ModelBase._exp_o_p(d, x0));
                expcexpdx2 = _ModelBase._exp_o_p(c,
                    _ModelBase._exp_o_p(d, points[2]));

                return (((points[3] * expcexpdx1) - (y0 * expcexpdx2))
                    / (expcexpdx1 - expcexpdx2));
              }
              return Double.NaN;
            }
          }
          break;
        }

        case 1: {// b
          switch (guesser) {
            case 0: {
              return _ModelBase._exp_o_p(-c, _ModelBase._exp_o_p(d, x0))
                  * (y0 - a);
            }
            case 1: {
              return (y0 - points[3])
                  / (_ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x0))
                      - _ModelBase._exp_o_p(c,
                          _ModelBase._exp_o_p(d, points[2])));
            }
          }
          break;
        }

        case 2: {// c
          switch (guesser) {
            case 0: {
              return _ModelBase._exp_o_p(-d, x0) * Math.log((y0 - a) / b);
            }

            case 1: {
              return _ModelBase._exp_o_p(-d, x0)
                  * Math.log((y0 / b) - (a / b));
            }
          }
          break;
        }

        case 3: {// d
          switch (guesser) {
            case 0: {
              return (Math.log(Math.log((y0 - a) / b) / c) / x0);
            }
            case 1: {
              return (Math.exp(Math.log((y0 / b) - (a / b)) / c) / x0);
            }
          }
          break;
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
            return ((newValue < -1e-13d) && (newValue > -1e100d));
          }
          case 2: {
            return ((newValue < -1e-13d) && (newValue > -1e5d));
          }
          case 3: {
            return ((newValue < -1e-13d) && (newValue > -1e5d));
          }
        }
      } else {
        switch (parameter) {
          case 1: {
            return ((newValue > 1e-13d) && (newValue < 1e100d));
          }
          case 2: {
            return ((newValue < -1e-13d) && (newValue > -1e5d));
          }
          case 3: {
            return ((newValue > 1e-13d) && (newValue < 1e5d));
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
        dest[0] = minMax[1];
        dest[1] = (minMax[0] - minMax[1]);

        steps = 100;
        do {
          temp = -40d * Math.exp(-14d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 2, temp, dest));
        dest[2] = temp;

        steps = 100;
        do {
          temp = -10d * Math.exp(-10d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 3, temp, dest));
        dest[3] = temp;

      } else {
        dest[0] = minMax[0];
        dest[1] = (minMax[1] - minMax[0]);

        steps = 100;
        do {
          temp = -10d * Math.exp(-10d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 2, temp, dest));
        dest[2] = temp;

        steps = 100;
        do {
          temp = 2d * Math.exp(-10d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 3, temp, dest));
        dest[3] = temp;
      }

      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return GompertzModel.this.value(x, parameters);
    }
  }
}
