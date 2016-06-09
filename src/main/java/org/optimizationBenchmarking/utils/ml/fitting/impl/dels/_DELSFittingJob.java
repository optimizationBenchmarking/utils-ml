package org.optimizationBenchmarking.utils.ml.fitting.impl.dels;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
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

  /** the maximum number of iterations for least squares methods */
  private static final int LEAST_SQUARES_MAX_ITERATIONS = 150;

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  protected _DELSFittingJob(final FittingJobBuilder builder) {
    super(builder);
    this.setLeastSquaresMaxIterations(
        _DELSFittingJob.LEAST_SQUARES_MAX_ITERATIONS);
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
    final int chosen;
    double factor;
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
    chosen = random.nextInt(dest.length);
    for (final double original : parent1Doubles) {
      ++index;
      factor = ((0.5d * random.nextDouble())
          + (0.1d * random.nextGaussian()));
      if ((index != chosen) && (random.nextBoolean())) {
        factor *= Math.exp(-random.nextInt(10) - 0.1d);
      }
      dest[index] = (original
          + (factor * (parent2Doubles[index] - parent3Doubles[index])));
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
      if (MathUtils.isFinite(solution.quality)) {
        return;
      }
    }
  }

  /**
   * Refine a solution for a random set of points
   *
   * @param solution
   *          the destination solution record
   * @param numPoints
   *          the number of points
   * @param random
   *          the random number generator
   */
  private final void __randomRefinement(
      final FittingCandidateSolution solution, final int numPoints,
      final Random random) {
    int[] selected;
    int index;

    index = solution.solution.length;
    selected = new int[index];
    for (; (--index) >= 0;) {
      selected[index] = random.nextInt(numPoints);
    }

    solution.quality = this.m_measure.evaluateAt(this.m_function,
        solution.solution, selected);
    this.selectPoints(selected);
    selected = null;

    this.refineWithLevenbergMarquardt(solution);
    this.selectPoints(null);
    solution.quality = this.evaluate(solution.solution);
  }

  /** {@inheritDoc} */
  @Override
  protected void doFit() {
    final int numParameters, numPoints, populationSize;
    final Random random;
    FittingCandidateSolution[] parents, offspring;
    FittingCandidateSolution current, parent1, parent2, parent3;
    IParameterGuesser guesser;
    int index, generation;

    // initialize and allocate all needed variables
    random = new Random();
    numParameters = this.m_function.getParameterCount();

    guesser = this.m_function.createParameterGuesser(this.m_data);

    numPoints = this.m_measure.getPointCount();

    populationSize = (numParameters * 45);

    parents = new FittingCandidateSolution[populationSize];
    offspring = new FittingCandidateSolution[populationSize];

    // initialize populations and generate first generation
    for (index = populationSize; (--index) >= 0;) {
      offspring[index] = new FittingCandidateSolution(numParameters);
      parents[index] = current = new FittingCandidateSolution(
          numParameters);
      this.__randomSolution(guesser, current, random);
      this.__randomRefinement(current, numPoints, random);
    }

    for (generation = (10 * populationSize); (--generation) >= 0;) {

      for (index = populationSize; (--index) >= 0;) {
        parent1 = parents[index];

        do {
          parent2 = parents[random.nextInt(populationSize)];
        } while (parent2 == parent1);
        do {
          parent3 = parents[random.nextInt(populationSize)];
        } while ((parent3 == parent2) || (parent3 == parent1));
        _DELSFittingJob.__deCrossover(parent1, parent2, parent3,
            offspring[index].solution, random);
      }

      for (index = populationSize; (--index) >= 0;) {
        current = offspring[index];
        parent1 = parents[index];
        if ((current.quality < parent1.quality)
            && MathUtils.isFinite(current.quality)) {
          parent1.assign(current.solution, current.quality);
        }
      }
    }

    current = offspring[0];
    offspring = parents = null;

    this.getCopyOfBest(current);
    this.setLeastSquaresMaxIterations(
        OptimizationBasedFittingJob.DEFAULT_LEAST_SQUARES_MAX_ITERATIONS);
    this.refineWithLeastSquaresAndSimplexSearch(current);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return DELSFitter.METHOD;
  }
}