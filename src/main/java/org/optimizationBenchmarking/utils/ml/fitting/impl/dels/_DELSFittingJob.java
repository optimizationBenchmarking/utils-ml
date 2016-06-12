package org.optimizationBenchmarking.utils.ml.fitting.impl.dels;

import java.util.Random;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingCandidateSolution;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;

/**
 * A function fitting job which mixes Differential Evolution, Least-Squares
 * optimization, and Simplex Search to obtain high-quality fittings. This
 * hybrid approach is intended to achieve better overall convergence.
 */
final class _DELSFittingJob
    extends OptimizationBasedFittingJob<FittingCandidateSolution> {

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  protected _DELSFittingJob(final FittingJobBuilder builder) {
    super(builder);
  }

  /**
   * Create one offspring with DE crossover
   *
   * @param parent1
   *          the first parent
   * @param parent2
   *          the second parent
   * @param parent3
   *          the third parent
   * @param dest
   *          the destination
   * @param random
   *          the random number generator
   */
  private static final void __deCrossover(
      final FittingCandidateSolution parent1,
      final FittingCandidateSolution parent2,
      final FittingCandidateSolution parent3, final double[] dest,
      final Random random) {
    final double[] parent1Doubles, parent2Doubles, parent3Doubles;
    // final int chosen;
    int index;

    parent1Doubles = parent1.solution;

    if (parent2.quality < parent3.quality) {
      parent2Doubles = parent2.solution;
      parent3Doubles = parent3.solution;
    } else {
      parent3Doubles = parent2.solution;
      parent2Doubles = parent3.solution;
    }

    // chosen = random.nextInt(parent1Doubles.length);
    index = (-1);
    for (final double original : parent1Doubles) {
      // ++index;
      // if ((index == chosen) || (random.nextInt(4) <= 0)) {
      // original = (original + (((0.5d * random.nextDouble())
      // + (0.1d * random.nextGaussian()))
      // * (parent2Doubles[index] - parent3Doubles[index])));
      // }
      ++index;
      dest[index] = original
          + (original + ((0.3d + (random.nextGaussian() * 0.1d))
              * (parent2Doubles[index] - parent3Doubles[index])));
    }
  }

  /**
   * Create one offspring with center crossover
   *
   * @param parent1
   *          the first parent
   * @param parent2
   *          the second parent
   * @param parent3
   *          the third parent
   * @param dest
   *          the destination
   * @param random
   *          the random number generator
   */
  private static final void __centerCrossover(
      final FittingCandidateSolution parent1,
      final FittingCandidateSolution parent2,
      final FittingCandidateSolution parent3, final double[] dest,
      final Random random) {
    final double[] parent1Doubles, parent2Doubles, parent3Doubles;
    int index;
    double weight1, weight2, mul;

    parent1Doubles = parent1.solution;

    if (parent2.quality < parent3.quality) {
      parent2Doubles = parent2.solution;
      parent3Doubles = parent3.solution;
    } else {
      parent3Doubles = parent2.solution;
      parent2Doubles = parent3.solution;
    }

    weight1 = (0.7d * random.nextDouble());
    weight2 = (0.7d * weight1 * random.nextDouble());
    mul = (1d / (weight1 + weight2 + 1d));

    index = (-1);
    for (final double original : parent1Doubles) {
      ++index;
      dest[index] = ((original + (weight1 * parent2Doubles[index])
          + (weight2 * parent3Doubles[index])) * mul);
    }
  }

  /**
   * Create a random solution
   *
   * @param guesser
   *          the parameter guesser
   * @param solution
   *          the destination solution record
   * @param random
   *          the random number generator
   */
  private final void __randomSolution(final IParameterGuesser guesser,
      final FittingCandidateSolution solution, final Random random) {
    int limiter;

    // make sure all points are valid
    for (limiter = 100; (--limiter) >= 0;) {
      guesser.createRandomGuess(solution.solution, random);
      solution.quality = this.evaluate(solution.solution);
      if ((solution.quality >= 0d)
          && (solution.quality < Double.POSITIVE_INFINITY)) {
        return;
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  protected void doFit() {
    final int numParameters, populationSize;
    final Random random;
    FittingCandidateSolution[] parents, offspring;
    FittingCandidateSolution current, parent1, parent2, parent3;
    IParameterGuesser guesser;
    int index, generation, maxIterations;

    // initialize and allocate all needed variables
    random = new Random();
    numParameters = this.m_function.getParameterCount();

    guesser = this.m_function.createParameterGuesser(this.m_data);

    populationSize = ((numParameters * numParameters) * 3);

    parents = new FittingCandidateSolution[populationSize];
    offspring = new FittingCandidateSolution[populationSize];

    maxIterations = this.getNumericalOptimizerMaxIterations();
    this.setNumericalOptimizerMaxIterations(
        numParameters * numParameters * numParameters);

    for (index = populationSize; (--index) >= 0;) {
      offspring[index] = new FittingCandidateSolution(numParameters);
      parents[index] = current = new FittingCandidateSolution(
          numParameters);
      this.__randomSolution(guesser, current, random);
      this.refineWithNelderMead(current);
    }

    for (generation = (numParameters * numParameters
        * populationSize); (--generation) >= 0;) {
      for (index = populationSize; (--index) >= 0;) {
        parent1 = parents[index];

        do {
          parent2 = parents[random.nextInt(populationSize)];
        } while (parent2 == parent1);
        do {
          parent3 = parents[random.nextInt(populationSize)];
        } while ((parent3 == parent2) || (parent3 == parent1));
        current = offspring[index];

        if (random.nextBoolean()) {
          _DELSFittingJob.__deCrossover(parent1, parent2, parent3,
              current.solution, random);
        } else {
          _DELSFittingJob.__centerCrossover(parent1, parent2, parent3,
              current.solution, random);
        }
        current.quality = this.value(current.solution);
      }

      for (index = populationSize; (--index) >= 0;) {
        current = offspring[index];
        parent1 = parents[index];
        if ((current.quality < parent1.quality)
            && (current.quality >= 0d)) {
          parent1.assign(current.solution, current.quality);
        }
      }
    }

    current = offspring[0];
    offspring = parents = null;

    this.getCopyOfBest(current);
    this.setNumericalOptimizerMaxIterations(maxIterations);
    this.refineWithLeastSquaresAndSimplexSearch(current);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return DELSFitter.METHOD;
  }
}