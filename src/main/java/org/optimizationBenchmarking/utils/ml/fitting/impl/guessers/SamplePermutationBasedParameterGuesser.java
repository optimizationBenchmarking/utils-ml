package org.optimizationBenchmarking.utils.ml.fitting.impl.guessers;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.combinatorics.PermutationIterator;
import org.optimizationBenchmarking.utils.math.functions.arithmetic.AddN;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;

/**
 * A parameter guesser which tries to compute the parameter values for each
 * permutation of the points from a sample.
 */
public abstract class SamplePermutationBasedParameterGuesser
    extends SampleBasedParameterGuesser {

  /** the current guess in progress */
  private final double[] m_currentGuess;

  /** the temporary point array */
  private final double[] m_tempPoints;

  /** the point permutations */
  private final PermutationIterator m_pointPermutations;

  /** the temporary array for errors */
  private final double[] m_errorTemp;

  /**
   * Create the parameter guesser
   *
   * @param data
   *          the data
   * @param parameterCount
   *          the number of parameters to guess
   * @param pointCount
   *          the number of required points
   */
  protected SamplePermutationBasedParameterGuesser(final IMatrix data,
      final int parameterCount, final int pointCount) {
    super(data, pointCount);

    this.m_currentGuess = new double[parameterCount];

    this.m_errorTemp = new double[pointCount];
    this.m_tempPoints = new double[pointCount << 1];
    this.m_pointPermutations = new PermutationIterator(pointCount, true);
  }

  /**
   * Create the parameter guesser
   *
   * @param data
   *          the data
   * @param parameterCount
   *          the number of parameters to guess
   */
  protected SamplePermutationBasedParameterGuesser(final IMatrix data,
      final int parameterCount) {
    this(data, parameterCount, parameterCount);
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
   * @return the error
   */
  protected final double error(final double[] points,
      final double[] parameters) {
    final double[] temp;
    double y, error;
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
      temp[i] = (y != 0d) ? error / Math.abs(y) : error;
    }
    return AddN.destructiveSum(temp);
  }

  /**
   * guess the parameter values based on a point permutation
   *
   * @param points
   *          the {@code (x,y)} value pairs
   * @param bestGuess
   *          the best current known guess
   * @param destGuess
   *          the destination array for the current guess
   */
  protected abstract void guessBasedOnPermutation(final double[] points,
      final double[] bestGuess, final double[] destGuess);

  /**
   * apply a permutation
   *
   * @param points
   *          the points
   * @param permutation
   *          the permutation
   * @param dest
   *          the destination
   */
  private static final void __applyPermutation(final double[] points,
      final int[] permutation, final double[] dest) {
    int index;

    index = (-1);
    for (int p : permutation) {
      p <<= 1;
      dest[++index] = points[p];
      dest[++index] = points[p + 1];
    }
  }

  /** {@inheritDoc} */
  @Override
  protected final boolean guess(final double[] points, final double[] dest,
      final Random random) {
    final double[] currentGuess, tempPoints;
    double newError, bestError;

    currentGuess = this.m_currentGuess; // the current guess
    tempPoints = this.m_tempPoints; // current permutation of the sample

    this.fallback(points, currentGuess, random);
    System.arraycopy(currentGuess, 0, dest, 0, currentGuess.length);
    bestError = this.error(points, currentGuess);

    // try all permutations of the points
    this.m_pointPermutations.reset();

    for (final int[] permutation : this.m_pointPermutations) {
      SamplePermutationBasedParameterGuesser.__applyPermutation(points,
          permutation, tempPoints); // apply the permutation

      // make a guess based on the current permutation of points
      this.guessBasedOnPermutation(tempPoints, dest, currentGuess);

      newError = this.error(points, currentGuess);
      if (newError >= 0d) {// is the guess feasible? (not NaN)
        // is the guess better than the best?
        if (newError < bestError) { // yes it is!
          System.arraycopy(currentGuess, 0, dest, 0, dest.length);
          if (newError <= 0d) {
            return true;
          }
          bestError = newError;
        }
      }
    }

    return MathUtils.isFinite(bestError);
  }
}