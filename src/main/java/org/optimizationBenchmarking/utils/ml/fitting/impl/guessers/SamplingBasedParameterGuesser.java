package org.optimizationBenchmarking.utils.ml.fitting.impl.guessers;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.functions.arithmetic.AddN;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;

/** A parameter guessing multiple times and returning the best result. */
public abstract class SamplingBasedParameterGuesser
    extends SampleBasedParameterGuesser {

  /** the current guess in progress */
  private final double[] m_currentGuess;

  /** the temporary array for errors */
  private final double[] m_errorTemp;

  /** the number of model "variants" */
  private final int m_variants;

  /**
   * Create the sample-based guesser. Normally, the number of rows in the
   * {@code data} matrix should be much higher than the number of
   * {@code required} points.
   *
   * @param data
   *          the data
   * @param variants
   *          the number of model "variants"
   * @param required
   *          the number of points needed for an educated guess
   * @param parameterCount
   *          the number of parameters
   */
  protected SamplingBasedParameterGuesser(final IMatrix data,
      final int variants, final int required, final int parameterCount) {
    super(data, required);

    this.m_variants = variants;
    this.m_currentGuess = new double[parameterCount];
    this.m_errorTemp = new double[required];
  }

  /**
   * Compute the value of the function to guess
   *
   * @param x
   *          the {@code x}-coordinate
   * @param parameters
   *          the parameters
   * @return the {@code y} value
   */
  protected abstract double value(final double x,
      final double[] parameters);

  /**
   * compute the error
   *
   * @param points
   *          the points
   * @param parameters
   *          the parameters
   * @param minAbs
   *          the minimum absolute {@code y} value
   * @return the error
   */
  private final double __error(final double[] points,
      final double[] parameters, final double minAbs) {
    final double[] temp;
    double error, y;
    int i, j;

    for (final double value : parameters) {
      if (!(MathUtils.isFinite(value))) {
        return Double.POSITIVE_INFINITY;
      }
    }

    temp = this.m_errorTemp;
    for (i = temp.length, j = (i << 1); (--i) >= 0;) {
      y = points[--j];
      error = Math.abs(y - this.value(points[--j], parameters));

      temp[i] = error / Math.max(0.9d * Math.abs(y), minAbs);
    }

    return AddN.destructiveSum(temp);
  }

  /**
   * guess if all required points are available
   *
   * @param variant
   *          the model variant
   * @param points
   *          an array with {@code x, y} coordinate pairs
   * @param dest
   *          the destination
   * @param random
   *          the random number generator
   * @return {@code true} if guessing was successful
   */
  protected boolean guess(final int variant, final double[] points,
      final double[] dest, final Random random) {
    return this.fallback(points, dest, random);
  }

  /** {@inheritDoc} */
  @Override
  protected final boolean guess(final double[] points, final double[] dest,
      final Random random) {
    final double[] currentGuess;
    final int variant;
    boolean hasGuess;
    double bestQuality, currentQuality, minAbs, minAbs2, currentAbs;
    int steps;

    currentGuess = this.m_currentGuess;
    hasGuess = false;
    bestQuality = Double.POSITIVE_INFINITY;

    variant = random.nextInt(this.m_variants);
    minAbs = minAbs2 = Double.POSITIVE_INFINITY;
    for (steps = points.length - 1; steps > 0; steps -= 2) {
      currentAbs = Math.abs(points[steps]);
      if (currentAbs < minAbs) {
        minAbs2 = minAbs;
        minAbs = currentAbs;
      } else {
        if (currentAbs < minAbs2) {
          minAbs2 = currentAbs;
        }
      }
    }

    if (minAbs <= 0d) {
      minAbs = minAbs2;
    }
    if ((minAbs <= 0d) || (minAbs >= Double.POSITIVE_INFINITY)) {
      minAbs = 1d;
    } else {
      minAbs2 = 0.9d * minAbs;
      if (minAbs2 > 0d) {
        minAbs = minAbs2;
      }
    }

    for (steps = (currentGuess.length
        * currentGuess.length); (--steps) >= 0;) {
      if (this.guess(variant, points, currentGuess, random)) {
        currentQuality = this.__error(points, currentGuess, minAbs);
        if ((!hasGuess) || (currentQuality < bestQuality)) {
          bestQuality = currentQuality;
          System.arraycopy(currentGuess, 0, dest, 0, dest.length);
        }
        hasGuess = true;
      }
    }
    return hasGuess;
  }
}
