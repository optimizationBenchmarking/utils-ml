package org.optimizationBenchmarking.utils.ml.fitting.impl.esls;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.primes.Primes;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;

/**
 * A function fitting job which mixes a traditional (mu+lambda)-ES and
 * Least-Squares optimization to obtain high-quality fittings.
 */
final class _ESLSFittingJob
    extends OptimizationBasedFittingJob<_ESCandidate> {

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  _ESLSFittingJob(final FittingJobBuilder builder) {
    super(builder);
  }

  /**
   * create an offspring solution from two parents
   *
   * @param parent1
   *          the first parent
   * @param parent2
   *          the second parent
   * @param stepLength
   *          the step length
   * @param random
   *          the random number generator
   * @param dest
   *          the destination solution
   */
  private static final void __recombineAndMutateSolution(
      final double[] parent1, final double[] parent2,
      final double[] stepLength, final Random random,
      final double[] dest) {
    int index;
    for (index = dest.length; (--index) >= 0;) {
      dest[index] = ((random.nextBoolean()// choose between discrete or
                                          // intermediate crossover
          ? (random.nextBoolean() ? parent1[index] : parent2[index])// discrete
          : (0.5d * (parent1[index] + parent2[index])))// intermediate
          + (random.nextGaussian() * stepLength[index]));// mutation
    }
  }

  /**
   * create an offspring step length from two parent step lengths
   *
   * @param parent1
   *          the first parent
   * @param parent2
   *          the second parent
   * @param tau0
   *          the tao-0 parameter
   * @param tau
   *          the tao parameter
   * @param random
   *          the random number generator
   * @param dest
   *          the destination solution
   */
  private static final void __recombineAndMutateStepLength(
      final double[] parent1, final double[] parent2, final double tau0,
      final double tau, final Random random, final double[] dest) {
    final double nu;
    int index;

    nu = Math.exp(tau0 * random.nextGaussian());
    for (index = dest.length; (--index) >= 0;) {
      dest[index] = ((0.5d * (parent1[index] + parent2[index]))// intermediate
          * nu * Math.exp(tau * random.nextGaussian()));// mutation
    }
  }

  /**
   * create an offspring solution from two parents
   *
   * @param parent1
   *          the first parent
   * @param parent2
   *          the second parent
   * @param tau0
   *          the tao-0 parameter
   * @param tau
   *          the tao parameter
   * @param random
   *          the random number generator
   * @param dest
   *          the destination solution
   */
  private static final void __createOffspring(final _ESCandidate parent1,
      final _ESCandidate parent2, final double tau0, final double tau,
      final Random random, final _ESCandidate dest) {
    _ESLSFittingJob.__recombineAndMutateStepLength(parent1.m_stepLength,
        parent2.m_stepLength, tau0, tau, random, dest.m_stepLength);
    _ESLSFittingJob.__recombineAndMutateSolution(parent1.solution,
        parent2.solution, dest.m_stepLength, random, dest.solution);
  }

  /** {@inheritDoc} */
  @Override
  protected final void doFit() {
    final int numParams, mu, lambda;
    final _ESCandidate[] population;
    final Random random;
    final double tau, tau0;
    IParameterGuesser guesser;
    _ESCandidate current, parent1, parent2;
    int generation, index, index2, findSamples;

    numParams = this.m_function.getParameterCount();
    mu = Primes.nextPrime((5 * (numParams + 1)) >>> 2);
    lambda = Primes.nextPrime(mu + 1);

    population = new _ESCandidate[mu + lambda];
    random = ThreadLocalRandom.current();

    guesser = this.m_function.createParameterGuesser(this.m_data);

    // create starting population of mu+lambda individuals
    for (index = population.length; (--index) >= 0;) {
      population[index] = current = new _ESCandidate(numParams);

      inner: for (findSamples = 100; (--findSamples) >= 0;) {
        guesser.createRandomGuess(current.solution, random);
        current.quality = this.evaluate(current.solution);
        if ((current.quality >= 0d)
            && (current.quality < Double.POSITIVE_INFINITY)) {
          break inner;
        }
      }

      // create step length
      for (index2 = numParams; (--index2) >= 0;) {
        current.m_stepLength[index2] = Math.max(1e-7d, //
            Math.abs(current.solution[index2] * random.nextFloat()
                * random.nextFloat() * random.nextFloat()));
      }
    }
    guesser = null;

    tau0 = 1d / Math.sqrt(numParams + numParams);
    tau = 1d / Math.sqrt(2d * Math.sqrt(numParams));

    // now perform actual algorithm
    this.setLeastSquaresMaxIterations(30);
    for (generation = Primes.nextPrime((7 * lambda) / 3); //
    (--generation) >= 0;) {

      // mu+lambda selection
      Arrays.sort(population);

      // generate new offsprings
      index2 = (-1);
      for (index = population.length; (--index) >= mu;) {
        current = population[index];

        // each parent has at least two offspring
        index2 = ((++index2) % mu);
        parent1 = population[index2];

        do {
          parent2 = population[random.nextInt(mu)];
        } while (parent2 == parent1);

        _ESLSFittingJob.__createOffspring(parent1, parent2, tau0, tau,
            random, current);
        current.quality = this.evaluate(current.solution);
      }

      if ((generation & 3) == 0) {
        // every 4 generations refine with least squares
        for (final _ESCandidate candidate : population) {
          this.refineWithLevenbergMarquardt(candidate);
        }
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return ESLSFitter.METHOD;
  }
}