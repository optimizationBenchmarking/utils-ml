package org.optimizationBenchmarking.utils.ml.fitting.impl.guessers;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.combinatorics.CanonicalPermutation;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;

/** A sampling-based parameter guesser which can improve its results. */
public abstract class ImprovingSamplingBasedParameterGuesser
    extends SamplingBasedParameterGuesser {

  /** the guessers */
  private final int[][] m_guessers;
  /** the guesser counts */
  private final int[] m_guesserCount;
  /** the parameters to check */
  private final int[] m_parameters;

  /**
   * Create the sample-based guesser. Normally, the number of rows in the
   * {@code data} matrix should be much higher than the number of
   * {@code required} points.
   *
   * @param data
   *          the data
   * @param variants
   *          the number of model "variants"
   * @param parameterCount
   *          the number of parameters
   * @param guesserCount
   *          the guesser count
   */
  protected ImprovingSamplingBasedParameterGuesser(final IMatrix data,
      final int variants, final int parameterCount,
      final int[] guesserCount) {
    super(data, variants, parameterCount);

    int index;

    if (guesserCount == null) {
      throw new IllegalArgumentException("Guesser list cannot be null."); //$NON-NLS-1$
    }
    if (guesserCount.length != parameterCount) {
      throw new IllegalArgumentException("Guesser list length must be " //$NON-NLS-1$
          + parameterCount + ", but is " + guesserCount.length); //$NON-NLS-1$
    }

    this.m_guesserCount = guesserCount;
    index = guesserCount.length;
    this.m_guessers = new int[index][];
    for (; (--index) >= 0;) {
      this.m_guessers[index] = CanonicalPermutation
          .createCanonicalZero(guesserCount[index]);
    }
    this.m_parameters = CanonicalPermutation
        .createCanonicalZero(parameterCount);
  }

  /**
   * Compute an improved parameter value
   *
   * @param variant
   *          the model variant
   * @param parameter
   *          the parameter index
   * @param guesser
   *          the guesser index
   * @param points
   *          the available points
   * @param parameters
   *          the current parameters values
   * @param random
   *          the random number generator
   * @return the new parameter value
   */
  protected double improveParameter(final int variant, final int parameter,
      final int guesser, final double[] points, final double[] parameters,
      final Random random) {
    throw new UnsupportedOperationException(("Parameter index " + parameter //$NON-NLS-1$
        + " does not support guesser index " + guesser + //$NON-NLS-1$
        " for variant " + variant) + '.');//$NON-NLS-1$
  }

  /**
   * Check whether a parameter value is acceptable
   *
   * @param variant
   *          the model variant
   * @param parameter
   *          the parameter index
   * @param newValue
   *          the new parameter value
   * @param parameters
   *          the current parameter values
   * @return {@code true} if the new parameter value is acceptable,
   *         {@code false} otherwise
   */
  protected boolean checkParameter(final int variant, final int parameter,
      final double newValue, final double[] parameters) {
    return MathUtils.isFinite(newValue);
  }

  /** {@inheritDoc} */
  @Override
  final boolean _doGuess(final int variant, final double[] points,
      final double[] dest, final Random random) {
    final int[] guesserCount, parameters;
    final int[][] guessers;
    int parameterCount, index, parameterIndex, parameterChoice,
        guesserIndex, guesserChoice, guesserNewCount;
    double newValue;
    boolean found;

    if (this.guess(variant, points, dest, random)) {

      guesserCount = this.m_guesserCount;
      guessers = this.m_guessers;
      parameters = this.m_parameters;

      do {

        index = (-1);
        for (final int[] guesser : guessers) {
          guesserCount[++index] = guesser.length;
        }
        parameterCount = parameters.length;

        found = false;
        parameterLoop: for (; parameterCount > 0;) {
          parameterIndex = random.nextInt(parameterCount);
          parameterChoice = parameters[parameterIndex];

          guesserNewCount = guesserCount[parameterChoice];
          if (guesserNewCount > 0) {
            guesserIndex = random.nextInt(guesserNewCount);
            guesserCount[parameterChoice] = (--guesserNewCount);
            guesserChoice = guessers[parameterChoice][guesserIndex];
            guessers[parameterChoice][guesserIndex] = guessers[parameterChoice][guesserNewCount];
            guessers[parameterChoice][guesserNewCount] = guesserChoice;

            newValue = this.improveParameter(variant, parameterChoice,
                guesserChoice, points, dest, random);
            if (this.checkParameter(variant, parameterChoice, newValue,
                dest)) {
              found = true;
              dest[parameterChoice] = newValue;
            } else {
              if (guesserNewCount > 0) {
                continue parameterLoop;
              }
            }
          }
          --parameterCount;
          parameters[parameterIndex] = parameters[parameterCount];
          parameters[parameterCount] = parameterChoice;
        }

      } while (found && random.nextBoolean());

      return true;
    }
    return false;
  }
}
