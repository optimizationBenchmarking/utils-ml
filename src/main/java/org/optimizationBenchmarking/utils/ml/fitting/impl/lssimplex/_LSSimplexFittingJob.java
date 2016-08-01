package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;

/**
 * A function fitting job to obtain high-quality fittings by first using an
 * ordinary least-squares problem approach (Levenberg-Marquardt or
 * Gauss-Newton algorithm) and then to refine the result using a direct
 * method, the Nelder-Mead simplex. As afterburner, we apply BOBYQA.
 */
final class _LSSimplexFittingJob
    extends OptimizationBasedFittingJob<_Candidate> {

  /** the maximum number of iterations for the main loop */
  private static final int MAIN_LOOP_ITERATIONS = 10;
  /** a similar solution has already been detected */
  private static final int RET_SAME = (OptimizationBasedFittingJob.RET_NO_IMPROVEMENT
      + 1);

  /**
   * the required steps for a solution to be considered sufficiently
   * different
   */
  private long m_steps;

  /** the candidate manager */
  private _CandidateManager m_manager;

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  _LSSimplexFittingJob(final FittingJobBuilder builder) {
    super(builder);
  }

  /** {@inheritDoc} */
  @Override
  protected final int checkImprovedSolution(final _Candidate solution) {
    return this.m_manager._isUniqueEnough(solution, this.m_steps)//
        ? OptimizationBasedFittingJob.RET_IMPROVEMENT
        : _LSSimplexFittingJob.RET_SAME;
  }

  /**
   * The final refinement step.
   *
   * @param solution
   *          the candidate solution to use
   */
  private final void __afterburner(final _Candidate solution) {
    double best;

    this.getCopyOfBest(solution);
    best = solution.quality;
    this.m_steps = 32L;
    if (this.refineWithBOBYQA(
        solution) != OptimizationBasedFittingJob.RET_IMPROVEMENT) {
      this.getCopyOfBest(solution);
      if (solution.quality > (0.999d * best)) {
        return;
      }
    }

    this.m_steps = 0L;
    this.m_manager.m_count = 0;
    this.refineWithLevenbergMarquardt(solution);
  }

  //// END: optimization routines

  /** {@inheritDoc} */
  @Override
  protected final void doFit() {
    final int numParameters, maxStartPointSamples;
    final Random random;
    final _Candidate bestSolution, tempSolution;
    IParameterGuesser guesser;
    int startPointIterations, mainIterations, initRetVal;
    boolean hasNoStart;

    // initialize and allocate all needed variables

    random = ThreadLocalRandom.current();
    numParameters = this.m_function.getParameterCount();

    this.m_manager = new _CandidateManager(numParameters,
        (_LSSimplexFittingJob.MAIN_LOOP_ITERATIONS * 16));

    guesser = this.m_function.createParameterGuesser(this.m_data);

    bestSolution = new _Candidate(numParameters);
    tempSolution = new _Candidate(numParameters);

    maxStartPointSamples = Math.max(10, Math.min(100, ((int) (Math.round(//
        2d * Math.pow(3d, numParameters)))))) / 3;

    // Inner loop: 1) generate initial guess, 2) use least-squares
    // approach to refine, 3) use direct black-box optimizer to refine,
    // 4) if that worked, try least-squares again
    for (mainIterations = 1; mainIterations <= _LSSimplexFittingJob.MAIN_LOOP_ITERATIONS; ++mainIterations) {

      // Find initial guess: we use the parameter guesser provided by the
      // model to create a few guesses and keep the best one
      hasNoStart = true;
      bestSolution.quality = Double.POSITIVE_INFINITY;
      startPointIterations = maxStartPointSamples;
      while (hasNoStart || ((--startPointIterations) >= 0)) {
        guesser.createRandomGuess(tempSolution.solution, random);
        this.subselect(numParameters, random);
        tempSolution.quality = this.value(tempSolution.solution);
        this.m_steps = 2048L;
        initRetVal = (random.nextBoolean() //
            ? this.refineWithLeastSquares(tempSolution)//
            : this.refineWithNelderMead(tempSolution));
        this.deselectPoints();

        if (initRetVal < _LSSimplexFittingJob.RET_SAME) {
          tempSolution.quality = this.evaluate(tempSolution.solution);
          if ((tempSolution.quality >= 0d) && (hasNoStart
              || (tempSolution.quality < bestSolution.quality))) {
            bestSolution._assign(tempSolution);
            hasNoStart = false;
          }
        }
      }
      this.m_manager._add(bestSolution);
      this.m_steps = 32L;
      this.refineWithLeastSquaresAndSimplexSearch(bestSolution);
    }

    this.__afterburner(bestSolution);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return LSSimplexFitter.METHOD;
  }
}
