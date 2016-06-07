package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

import java.util.Random;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.fitting.leastsquares.GaussNewtonOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.apache.commons.math3.util.Incrementor;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;

/**
 * A function fitting job which proceeds as follows:
 * <p>
 * The goal is to obtain high-quality fittings by first using an ordinary
 * least-squares problem approach (Levenberg-Marquardt or Gauss-Newton
 * algorithm) and then to refine the result using a direct method, the
 * Nelder-Mead simplex.
 * </p>
 * <p>
 * Besides getting high-quality results, having stable and reproducible
 * results is also very important. This is why we perform the above several
 * times, more often if results seem to be unstable.
 * </p>
 */
final class _LSSimplexFittingJob extends FittingJob
    implements MultivariateFunction, LeastSquaresProblem,
    ConvergenceChecker<Evaluation> {

  /** Relative tolerance threshold. */
  private static final double OPTIMIZER_RELATIVE_THRESHOLD = 1e-10d;

  /** the maximum number of iterations */
  private static final int OPTIMIZER_MAX_ITERATIONS = 768;

  /** the maximum number of iterations for the main loop */
  private static final int MAIN_LOOP_ITERATIONS = 10;

  /** the solution was improved */
  private static final int RET_IMPROVEMENT = 0;
  /** the application of the optimization method has failed */
  private static final int RET_FAILED = 1;
  /** the solution was not improved */
  private static final int RET_NO_IMPROVEMENT = 2;
  /** a similar solution has already been detected */
  private static final int RET_SAME = 3;

  /** the evaluation counter */
  private Incrementor m_evaluationCounter;
  /** the iteration counter */
  private Incrementor m_iterationCounter;

  /** the start vector */
  private ArrayRealVector m_startVector;

  /** the start vector data */
  private double[] m_startVectorData;

  /** the Gauss-Newton optimizer */
  private GaussNewtonOptimizer m_gaussNewton;

  /** the Levenberg-Marquardt optimizer */
  private LevenbergMarquardtOptimizer m_levenbergMarquardt;
  /** the objective function */
  private ObjectiveFunction m_objective;
  /** the maximum evaluations */
  private MaxEval m_maxEval;

  /** the maximum iterations */
  private MaxIter m_maxIter;

  /** the simplex optimizer */
  private SimplexOptimizer m_simplex;

  /** the candidate manager */
  private _CandidateManager m_manager;

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  protected _LSSimplexFittingJob(final FittingJobBuilder builder) {
    super(builder);
  }

  //// BEGIN: basic functions of the implemented interfaces

  /** {@inheritDoc} */
  @Override
  public final RealVector getStart() {
    return this.m_startVector;
  }

  /** {@inheritDoc} */
  @Override
  public final int getObservationSize() {
    return this.m_data.m();
  }

  /** {@inheritDoc} */
  @Override
  public final int getParameterSize() {
    return this.m_function.getParameterCount();
  }

  /**
   * get the {@code double[]} associated with a real vector
   *
   * @param vec
   *          the vector
   * @return the {@code double[]}
   */
  private static final double[] __toArray(final RealVector vec) {
    return ((vec instanceof ArrayRealVector)//
        ? ((ArrayRealVector) vec).getDataRef() : vec.toArray());
  }

  /** {@inheritDoc} */
  @Override
  public final Evaluation evaluate(final RealVector point) {
    final _InternalEvaluation eval;
    final double[] vector;

    eval = new _InternalEvaluation(point);
    vector = _LSSimplexFittingJob.__toArray(point);
    this.m_measure.evaluate(this.m_function, vector, eval);
    this.register(eval.quality, vector);

    eval.m_jacobian = new Array2DRowRealMatrix(eval.jacobian, false);
    eval.m_residuals = new ArrayRealVector(eval.residuals, false);
    return eval;
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double[] point) {
    return this.evaluate(point);
  }

  /** {@inheritDoc} */
  @Override
  public final boolean converged(final int iteration,
      final Evaluation previous, final Evaluation current) {
    final RealVector pv, cv;
    final double[] p, c;
    double ci;
    int i;

    if (iteration >= _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS) {
      return true;
    }

    pv = previous.getPoint();
    cv = current.getPoint();

    p = _LSSimplexFittingJob.__toArray(pv);
    c = _LSSimplexFittingJob.__toArray(cv);

    i = (-1);
    for (final double ppi : p) {
      ci = c[++i];
      if (Math.abs(
          ppi - ci) > (_LSSimplexFittingJob.OPTIMIZER_RELATIVE_THRESHOLD
              * Math.max(Math.abs(ppi), Math.abs(ci)))) {
        return false;
      }
    }

    return true;
  }

  /** {@inheritDoc} */
  @Override
  public final Incrementor getEvaluationCounter() {
    return this.m_evaluationCounter;
  }

  /** {@inheritDoc} */
  @Override
  public final Incrementor getIterationCounter() {
    return this.m_iterationCounter;
  }

  /** {@inheritDoc} */
  @Override
  public final ConvergenceChecker<Evaluation> getConvergenceChecker() {
    return this;
  }
  //// END: basic functions of the implemented interfaces

  //// BEGIN: optimization routines
  /**
   * Refine the current {@link #m_startVector start point} with the
   * Levenberg-Marquardt method.
   *
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  private final int __refineStartWithLevenbergMarquardt(
      final _Candidate solution, final long steps) {
    try {
      this.m_iterationCounter = new Incrementor(
          _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);
      this.m_evaluationCounter = new Incrementor(
          _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS
              * _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);

      if (this.m_levenbergMarquardt == null) {
        this.m_levenbergMarquardt = new LevenbergMarquardtOptimizer();
      }

      return this.__checkOptimum(this.m_levenbergMarquardt.optimize(this),
          solution, steps);
    } catch (@SuppressWarnings("unused") final Throwable error) {
      return _LSSimplexFittingJob.RET_FAILED;
    } finally {
      this.m_evaluationCounter = null;
      this.m_iterationCounter = null;
    }
  }

  /**
   * Check an optimum
   *
   * @param optimum
   *          the discovered optimum
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  private final int __checkOptimum(final Optimum optimum,
      final _Candidate solution, final long steps) {
    final double quality;
    quality = optimum.getRMS();

    if ((quality < solution.quality) && MathUtils.isFinite(quality)) {
      solution._assign(_LSSimplexFittingJob.__toArray(optimum.getPoint()),
          solution.quality);
      this.register(solution.quality, solution.solution);
      if (this.m_manager._isUniqueEnough(solution, steps)) {
        this.m_manager._add(solution);
        return _LSSimplexFittingJob.RET_IMPROVEMENT;
      }
      return _LSSimplexFittingJob.RET_SAME;
    }
    return _LSSimplexFittingJob.RET_NO_IMPROVEMENT;
  }

  /**
   * refine the current {@link #m_startVector start point} with the
   * Gauss-Newton method.
   *
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  private final int __refineStartWithGaussNewton(final _Candidate solution,
      final long steps) {
    try {
      this.m_iterationCounter = new Incrementor(
          _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);
      this.m_evaluationCounter = new Incrementor(
          _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS
              * _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);

      if (this.m_gaussNewton == null) {
        this.m_gaussNewton = new GaussNewtonOptimizer(
            GaussNewtonOptimizer.Decomposition.SVD);
      }

      return this.__checkOptimum(this.m_gaussNewton.optimize(this),
          solution, steps);
    } catch (@SuppressWarnings("unused") final Throwable error) {
      return _LSSimplexFittingJob.RET_FAILED;
    } finally {
      this.m_evaluationCounter = null;
      this.m_iterationCounter = null;
    }
  }

  /**
   * Refine the given solution with a least-squares method
   *
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  private final int __refineWithLeastSquares(final _Candidate solution,
      final long steps) {
    System.arraycopy(solution.solution, 0, this.m_startVectorData, 0,
        this.m_startVectorData.length);

    switch (this.__refineStartWithLevenbergMarquardt(solution, steps)) {
      case RET_IMPROVEMENT: {
        return _LSSimplexFittingJob.RET_IMPROVEMENT;
      }
      case RET_FAILED:
      case RET_NO_IMPROVEMENT: {
        return this.__refineStartWithGaussNewton(solution, steps);
      }
      default: {
        return _LSSimplexFittingJob.RET_SAME;
      }
    }
  }

  /**
   * refine a given solution using Nelder-Mead
   *
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  @SuppressWarnings("unused")
  private final int __refineWithNelderMead(final _Candidate solution,
      final long steps) {
    final int dim;

    try {
      if (this.m_simplex == null) {
        this.m_simplex = new SimplexOptimizer(1e-10d,
            Double.NEGATIVE_INFINITY);
      }
      if (this.m_objective == null) {
        this.m_objective = new ObjectiveFunction(this);
      }
      dim = solution.solution.length;

      if (this.m_maxEval == null) {
        this.m_maxEval = new MaxEval(dim * dim * 300);
      }
      if (this.m_maxIter == null) {
        this.m_maxIter = new MaxIter(this.m_maxEval.getMaxEval());
      }

      return this.__checkPointValuePair(this.m_simplex.optimize(//
          new NelderMeadSimplex(solution.solution), //
          new InitialGuess(solution.solution), //
          this.m_objective, this.m_maxEval, this.m_maxIter,
          GoalType.MINIMIZE), solution, steps);

    } catch (final Throwable error) {
      return _LSSimplexFittingJob.RET_FAILED;
    }
  }

  /**
   * Check an point-value pair optimum
   *
   * @param optimum
   *          the discovered optimum
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  private final int __checkPointValuePair(final PointValuePair optimum,
      final _Candidate solution, final long steps) {
    final double quality;

    quality = optimum.getValue().doubleValue();

    if (MathUtils.isFinite(quality)) {
      if (quality < solution.quality) {
        solution._assign(optimum.getPoint(), solution.quality);
        this.register(solution.quality, solution.solution);
        if (this.m_manager._isUniqueEnough(solution, steps)) {
          this.m_manager._add(solution);
          return _LSSimplexFittingJob.RET_IMPROVEMENT;
        }
        return _LSSimplexFittingJob.RET_SAME;
      }
    }
    return _LSSimplexFittingJob.RET_NO_IMPROVEMENT;
  }

  /**
   * refine a given solution using all available means
   *
   * @param solution
   *          the solution to refine
   * @param steps
   *          the number of steps required between at least two variables
   * @return one of the {@code RET_} codes
   */
  @SuppressWarnings("incomplete-switch")
  private final int __refine(final _Candidate solution, final long steps) {
    boolean doLocalSearch, hasImprovement;

    doLocalSearch = true;
    hasImprovement = false;
    loop: for (;;) {
      switch (this.__refineWithLeastSquares(solution, steps)) {
        case RET_IMPROVEMENT: {
          hasImprovement = doLocalSearch = true;
          break;
        }
        case RET_SAME: {
          return _LSSimplexFittingJob.RET_SAME;
        }
      }
      if (doLocalSearch) {
        switch (this.__refineWithNelderMead(solution, steps)) {
          case RET_IMPROVEMENT: {
            hasImprovement = true;
            doLocalSearch = false;
            continue loop;
          }
          case RET_SAME: {
            return _LSSimplexFittingJob.RET_SAME;
          }
        }
      }
      if (hasImprovement) {
        return _LSSimplexFittingJob.RET_IMPROVEMENT;
      }
      return _LSSimplexFittingJob.RET_NO_IMPROVEMENT;
    }
  }

  //// END: optimization routines

  /** {@inheritDoc} */
  @Override
  protected void fit() {
    final int numParameters, maxStartPointSamples;
    final Random random;
    _Candidate bestSolution, tempSolution;
    IParameterGuesser guesser;
    int index, iterations;
    boolean hasNoStart;

    // initialize and allocate all needed variables

    random = new Random();
    numParameters = this.m_function.getParameterCount();

    this.m_manager = new _CandidateManager(numParameters,
        (_LSSimplexFittingJob.MAIN_LOOP_ITERATIONS * 16));

    guesser = this.m_function.createParameterGuesser(this.m_data);

    bestSolution = new _Candidate(numParameters);
    tempSolution = new _Candidate(numParameters);
    this.m_startVectorData = tempSolution.solution;
    this.m_startVector = new ArrayRealVector(this.m_startVectorData,
        false);

    maxStartPointSamples = Math.max(100,
        Math.min(10000, ((int) (Math.round(//
            2d * Math.pow(3d, numParameters))))))
        / 3;

    // Inner loop: 1) generate initial guess, 2) use least-squares
    // approach to refine, 3) use direct black-box optimizer to refine,
    // 4) if that worked, try least-squares again
    for (iterations = 1; iterations <= _LSSimplexFittingJob.MAIN_LOOP_ITERATIONS; ++iterations) {

      // Find initial guess: we use the parameter guesser provided by the
      // model to create a few guesses and keep the best one
      hasNoStart = true;
      bestSolution.quality = Double.POSITIVE_INFINITY;
      for (index = maxStartPointSamples; (--index) >= 0;) {
        guesser.createRandomGuess(tempSolution.solution, random);
        tempSolution.quality = this.evaluate(tempSolution.solution);
        if (MathUtils.isFinite(tempSolution.quality) && (hasNoStart
            || (tempSolution.quality < bestSolution.quality))) {
          // Try to get starting points which are, sort of, different.
          tempSolution._assign(tempSolution.solution,
              tempSolution.quality);
          if (this.m_manager._isUniqueEnough(tempSolution, 2048L)) {
            bestSolution._assign(tempSolution);
            hasNoStart = false;
          }
        }
      }
      if (hasNoStart) {
        guesser.createRandomGuess(bestSolution.solution, random);
        bestSolution._assign(bestSolution.solution,
            this.evaluate(tempSolution.solution));
      }
      this.m_manager._add(bestSolution);

      this.__refine(bestSolution, 32);
    }

    // Dispose all the variables.
    this.m_gaussNewton = null;
    this.m_levenbergMarquardt = null;
    this.m_startVector = null;
    this.m_simplex = null;
    this.m_objective = null;
    this.m_maxEval = null;
    this.m_maxIter = null;
    this.m_startVectorData = null;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return LSSimplexFitter.METHOD;
  }
}
