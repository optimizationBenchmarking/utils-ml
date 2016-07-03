package org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.statistics.aggregate.StandardDeviationAggregate;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingCandidateSolution;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;

/**
 * A function fitting job which mixes CMA-ES and Least-Squares optimization
 * to obtain high-quality fittings. This hybrid approach is intended to
 * achieve better overall convergence.
 */
final class _CMAESLSFittingJob
    extends OptimizationBasedFittingJob<FittingCandidateSolution> {

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  protected _CMAESLSFittingJob(final FittingJobBuilder builder) {
    super(builder);
  }

  /** {@inheritDoc} */
  @Override
  protected void doFit() {
    final int numParams;
    final FittingCandidateSolution current;
    final int maxLSIterations;
    Random random;
    IParameterGuesser guesser;
    StandardDeviationAggregate[] stddev;
    double[] stddevs, currentArray;
    int index, samples, subsamples;
    double quality;

    numParams = this.m_function.getParameterCount();
    current = new FittingCandidateSolution(numParams);
    currentArray = current.solution;
    stddev = new StandardDeviationAggregate[numParams];
    stddevs = new double[numParams];
    random = new Random();

    for (index = numParams; (--index) >= 0;) {
      stddev[index] = new StandardDeviationAggregate();
    }

    maxLSIterations = this.getLeastSquaresMaxIterations();
    this.setLeastSquaresMaxIterations(((numParams * numParams) * 8) / 5);

    guesser = this.m_function.createParameterGuesser(this.m_data);
    for (samples = ((numParams * 7) / 4) + 1; (--samples) >= 0;) {

      current.quality = Double.POSITIVE_INFINITY;
      for (subsamples = (10 * numParams); (--subsamples) >= 0;) {
        guesser.createRandomGuess(stddevs, random);
        quality = this.evaluate(stddevs);
        if ((quality > 0d) && (quality < current.quality)) {
          current.assign(stddevs, quality);
        }
      }

      this.refineWithLevenbergMarquardt(current);
      for (index = numParams; (--index) >= 0;) {
        stddev[index].append(currentArray[index]);
      }
    }

    for (index = numParams; (--index) >= 0;) {
      stddevs[index] = stddev[index].doubleValue();
    }
    stddev = null;

    this.setNumericalOptimizerMaxIterations(numParams * numParams * 700);
    this.setLeastSquaresMaxIterations(maxLSIterations);
    this.getCopyOfBest(current);

    this.refineWithCMAES(current, stddevs);
    this.refineWithLevenbergMarquardt(current);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return CMAESLSFitter.METHOD;
  }
}