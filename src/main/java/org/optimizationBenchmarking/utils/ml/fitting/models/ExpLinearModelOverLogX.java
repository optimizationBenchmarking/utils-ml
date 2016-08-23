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
        ._exp_o_p(parameters[2], Math.log(parameters[3] + x))));
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
    gradient[2] = _ModelBase._gradient(b * dxc * Math.log(dx), c);
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

  /** the parameter guesser */
  private final class __ExpLinearModelOverLogXParameterGuesser
      extends ImprovingSamplingBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __ExpLinearModelOverLogXParameterGuesser(final IMatrix data) {
      super(data, 2, 4, new int[] { 4, 4, 2, 1, });
    }

    /** {@inheritDoc} */
    @SuppressWarnings("incomplete-switch")
    @Override
    protected final double improveParameter(final int variant,
        final int parameter, final int guesser, final double[] points,
        final double[] parameters, final Random random) {
      final double x0, y0, a, b, c, d, dx0c, dx1c, expclogx0d, expclogx1d;

      x0 = points[0];
      y0 = points[1];
      a = parameters[0];
      b = parameters[1];
      c = parameters[2];
      d = parameters[3];

      switch (parameter) {
        case 0: {
          switch (guesser) {
            case 0: {
              return y0 - (b * Math.exp(c * Math.log(x0 + d)));
            }
            case 1: {
              return y0 - (b * _ModelBase._pow((d + x0), c));
            }
            case 2: {
              if (points.length < 4) {
                return Double.NaN;
              }
              dx0c = _ModelBase._pow(d + x0, c);
              dx1c = _ModelBase._pow(d + points[2], c);

              return ((points[3] * dx0c) - (y0 * dx1c)) / (dx0c - dx1c);
            }
            case 3: {
              if (points.length < 4) {
                return Double.NaN;
              }
              expclogx0d = Math.exp(c * Math.log(x0 + d));
              expclogx1d = Math.exp(c * Math.log(points[2] + d));

              return ((y0 * expclogx1d) - (points[3] * expclogx0d))
                  / (expclogx1d - expclogx0d);
            }
          }
          break;
        }
        case 1: {
          switch (guesser) {
            case 0: {
              return Math.exp(-(c * Math.log(x0 + d))) * (y0 - a);
            }
            case 1: {
              return Math.exp(-(c * Math.log(x0 + d))) * (y0 - a);
            }
            case 2: {
              if (points.length < 4) {
                return Double.NaN;
              }
              return (points[3] - y0)
                  / (Math.exp(c * Math.log(points[2] + d))
                      - Math.exp(c * Math.log(x0 + d)));
            }
            case 3: {
              if (points.length < 4) {
                return Double.NaN;
              }
              return (points[3] - y0) / (_ModelBase._pow(d + x0, c)
                  - _ModelBase._pow(d + points[2], c));
            }
          }
          break;
        }
        case 2: {
          switch (guesser) {
            case 0: {
              return (Math.log((y0 - a) / b)) / (Math.log(d + x0));
            }
            case 1: {
              if (points.length < 4) {
                return Double.NaN;
              }
              return (Math.log(y0 - a) - Math.log(points[3] - a))
                  / (Math.log(d + x0) - Math.log(d + points[2]));
            }
          }
          break;
        }
        case 3: {
          return _ModelBase._pow((-(a - y0) / b), (1d / c)) - x0;
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
            return ((newValue < -1e-13d) && (newValue > -1e2d));
          }
          case 3: {
            return ((newValue > (-this.m_minX)) && (newValue < 1e3d));
          }
        }
      } else {
        switch (parameter) {
          case 1: {
            return ((newValue < -1e-13d) && (newValue > -1e100d));
          }
          case 2: {
            return ((newValue > 1e-13d) && (newValue < 1e2d));
          }
          case 3: {
            return ((newValue > (-this.m_minX)) && (newValue < 1e3d));
          }
        }
      }

      return ((newValue > -1e100d) && (newValue < 1e100d));
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean guess(final int variant, final double[] points,
        final double[] dest, final Random random) {
      final double[] minMaxY, minMaxX;
      double temp;
      int steps;

      minMaxY = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);
      minMaxX = _ModelBase._getMinMax(false, this.m_minX, this.m_maxX,
          points, random);

      if (variant == 0) {
        dest[0] = minMaxY[0];
        dest[1] = minMaxY[1] - minMaxY[0];

        steps = 100;
        do {
          temp = -3d * Math.exp(-13d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 2, temp, dest));
        dest[2] = temp;

      } else {
        dest[0] = minMaxY[1];
        dest[1] = minMaxY[0] - minMaxY[1];

        steps = 100;
        do {
          temp = -(Math.exp(
              random.nextDouble() * Math.log(minMaxY[1] - minMaxY[0])));
        } while (((--steps) > 0) && this.checkParameter(0, 1, temp, dest));
        dest[1] = temp;

        steps = 100;
        do {
          temp = 2d * Math.exp(-3.5d * random.nextDouble());
        } while (((--steps) > 0) && this.checkParameter(0, 2, temp, dest));
        dest[2] = temp;

      }

      steps = 100;
      do {
        temp = -(minMaxX[0] + Math
            .exp(random.nextDouble() * Math.log(minMaxX[1] - minMaxX[0])));
      } while (((--steps) > 0) && this.checkParameter(0, 3, temp, dest));
      dest[3] = temp;

      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return ExpLinearModelOverLogX.this.value(x, parameters);
    }
  }
}