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

  /** the checker for {@code a} */
  static final ParameterValueCheckerMinMax A = new ParameterValueCheckerMinMax(
      -1e30d, 1e30d);
  /** the checker for {@code b} */
  static final ParameterValueCheckerMinMax B = GompertzModel.A;
  /** the checker for {@code c} */
  static final ParameterValueCheckerMinMaxAbs C = new ParameterValueCheckerMinMaxAbs(
      1e-6d, 1e4d);
  /** the checker for {@code d} */
  static final ParameterValueCheckerMinMaxAbs D = GompertzModel.C;

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
      expdx = _ModelBase._exp(dx);
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
    gradient[1] = _ModelBase._gradient(_ModelBase._exp(cexpdx), b);
    if ((b != 0d) && (b == b)) {
      cexpdx = (b * _ModelBase._exp(cexpdx + dx));
      gradient[2] = _ModelBase._gradient(cexpdx, c);
      gradient[3] = _ModelBase._gradient((cexpdx * c * x), d);
    }
    gradient[2] = gradient[3] = 0d;
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
    double trialA, trialB;
    int trials;

    trials = 100;
    do {
      trialA = (minY * (1d + (0.05d * random.nextGaussian())));
    } while ((((minY > 0d) && (trialA < 0d)) || (trialA >= maxY))
        && ((--trials) >= 0));

    trials = 100;
    do {
      trialB = ((maxY - trialA) * ((1d + //
          Math.abs(0.05d * random.nextGaussian()))));
    } while (((minY > 0d) && ((maxY - trialB) < 0d)) && ((--trials) >= 0));

    dest[0] = trialA + trialB;
    dest[1] = -trialB;

    dest[2] = -(random.nextDouble() + random.nextInt(5));
    dest[3] = -(random.nextDouble() + random.nextInt(5));
  }

  /**
   * compute {@code a} from {@code c} and {@code d} and two points
   * according to
   * {@code a=-(exp(c*exp(d*x2))*y1-exp(c*exp(d*x1))*y2)/(exp(c*exp(d*x1))-exp(c*exp(d*x2)))}
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
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return {@code a}
   */
  static final double _a_x1y1x2y2cd(final double x1, final double y1,
      final double x2, final double y2, final double c, final double d) {
    final double expcexpdx1, expcexpdx2;

    expcexpdx1 = _ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x1));
    expcexpdx2 = _ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x2));

    return (((y2 * expcexpdx1) - (y1 * expcexpdx2))
        / (expcexpdx1 - expcexpdx2));
  }

  /**
   * compute {@code b} from {@code c} and {@code d} and two points
   * according to {@code b=(y1-y2)/(exp(c*exp(d*x1))-exp(c*exp(d*x2)))}
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
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return {@code b}
   */
  static final double _b_x1y1x2y2cd(final double x1, final double y1,
      final double x2, final double y2, final double c, final double d) {
    return (y1 - y2) / (_ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x1))
        - _ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x2)));
  }

  /**
   * compute {@code a} from {@code b}, {@code c}, and {@code d} and two
   * points according to {@code a=y1-b*exp(c*exp(d*x1))}
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return {@code a}
   */
  static final double _a_x1y1bcd(final double x1, final double y1,
      final double b, final double c, final double d) {
    return (y1 - (b * _ModelBase._exp_o_p(c, _ModelBase._exp_o_p(d, x1))));
  }

  /**
   * compute {@code b} from {@code a}, {@code c}, and {@code d} and two
   * points according to {@code b=exp(-(c*exp(d*x1)))*(y1-a)}
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the value of {@code a}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return {@code b}
   */
  static final double _b_x1y1acd(final double x1, final double y1,
      final double a, final double c, final double d) {
    return _ModelBase._exp_o_p(-c, _ModelBase._exp_o_p(d, x1)) * (y1 - a);
  }

  /**
   * compute {@code c} from {@code a}, {@code b}, and {@code d} and two
   * points according to {@code c=exp(-(d*x1))*log(y1/b-a/b)}
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param d
   *          the value of {@code d}
   * @return {@code c}
   */
  static final double _c_x1y1abd(final double x1, final double y1,
      final double a, final double b, final double d) {
    return ParameterValueChecker.choose(//
        _ModelBase._exp_o_p(-d, x1) * _ModelBase._log((y1 - a) / b), //
        _ModelBase._exp_o_p(-d, x1) * _ModelBase._log((y1 / b) - (a / b)), //
        GompertzModel.C);
  }

  /**
   * compute {@code c} from {@code a}, {@code b}, and {@code c} and two
   * points according to {@code d=log(log(y1/b-a/b)/c)/x1)}
   *
   * @param x1
   *          the {@code x}-coordinate of the first point
   * @param y1
   *          the {@code y}-coordinate of the first point
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @return {@code d}
   */
  static final double _d_x1y1abc(final double x1, final double y1,
      final double a, final double b, final double c) {
    return ParameterValueChecker.choose(//
        (_ModelBase._log(_ModelBase._log((y1 - a) / b) / c) / x1), //
        (_ModelBase._log(_ModelBase._log((y1 / b) - (a / b)) / c) / x1), //
        GompertzModel.D);
  }

  /** the parameter guesser */
  private class __GompertzModelParameterGuesser
      extends SamplePermutationBasedParameterGuesser {

    /**
     * create the model
     *
     * @param data
     *          the data
     */
    __GompertzModelParameterGuesser(final IMatrix data) {
      super(data, GompertzModel.this.getParameterCount(),
          GompertzModel.this.getParameterCount() - 1);
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return GompertzModel.this.value(x, parameters);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean fallback(final double[] points,
        final double[] dest, final Random random) {
      double minY, maxY;
      int i;

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

      GompertzModel._fallback(minY, maxY, dest, random);
      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      final double maxY, minY;

      minY = (random.nextBoolean() ? this.m_minY
          : ((random.nextInt(11) - 5) + (random.nextGaussian() * 10)));
      maxY = (random.nextBoolean() ? this.m_maxY
          : (minY
              + Math.abs(random.nextInt(1000) * random.nextGaussian())));
      GompertzModel._fallback(minY, maxY, dest, random);
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

            newA = GompertzModel._a_x1y1x2y2cd(points[0], points[1],
                points[2], points[3], (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (GompertzModel.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }

            newA = GompertzModel._a_x1y1bcd(points[0], points[1],
                (hasB ? newB : oldB), (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (GompertzModel.A.check(newA)) {
              changed = hasA = true;
              break findA;
            }
          }
        }

        // find B
        findB: {
          if (!hasB) {
            newB = GompertzModel._b_x1y1x2y2cd(points[0], points[1],
                points[2], points[3], (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (GompertzModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }

            newB = GompertzModel._b_x1y1acd(points[0], points[1],
                (hasA ? newA : oldA), (hasC ? newC : oldC),
                (hasD ? newD : oldD));
            if (GompertzModel.B.check(newB)) {
              changed = hasB = true;
              break findB;
            }
          }
        }

        // find C
        findC: {
          if (!hasC) {
            newC = GompertzModel._c_x1y1abd(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasD ? newD : oldD));
            if (GompertzModel.C.check(newC)) {
              changed = hasC = true;
              break findC;
            }
          }
        }

        // find D
        findD: {
          if (!hasD) {
            newD = GompertzModel._d_x1y1abc(points[0], points[1],
                (hasA ? newA : oldA), (hasB ? newB : oldB),
                (hasC ? newC : oldC));
            if (GompertzModel.D.check(newD)) {
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
              if (GompertzModel.A.check(newA)) {
                hasA = changed = true;
                break emergency;
              }
            }

            if (!(hasB)) {
              newB = ((hasA ? newA
                  : Math.min(points[1], Math.min(points[3], points[5]))) - //
                  Math.max(points[1], Math.max(points[3], points[5])));
              if (GompertzModel.B.check(newB)) {
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
