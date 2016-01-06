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

  /** the minimum required weighted distance between starting points */
  private static final double MIN_REQUIRED_DISTANCE = 1e-5d;

  /** a solution was used as input to levenberg-marquardt */
  private static final int PROCESSED_BY_LEVENBERG_MARQUARDT = 1;
  /** a solution was used as input to gauss-newton */
  private static final int PROCESSED_BY_GAUSS_NEWTON = //
  (_LSSimplexFittingJob.PROCESSED_BY_LEVENBERG_MARQUARDT << 1);
  /** a solution was used as input to nelder-mead */
  private static final int PROCESSED_BY_NELDER_MEAD = //
  (_LSSimplexFittingJob.PROCESSED_BY_GAUSS_NEWTON << 1);

  /** the solution was used as input for any of least-squares solvers */
  private static final int PROCESSED_BY_LEAST_SQUARES = //
  (_LSSimplexFittingJob.PROCESSED_BY_LEVENBERG_MARQUARDT
      | _LSSimplexFittingJob.PROCESSED_BY_GAUSS_NEWTON);

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
   * @param source
   *          the source candidate solution
   * @param dest
   *          the destination candidate solution
   * @return {@code true} on success, {@code false} on failure
   */
  private final boolean __refineStartWithLevenbergMarquardt(
      final _Candidate source, final _Candidate dest) {
    final Optimum res;
    final double quality;

    if ((source.m_processedBy
        & _LSSimplexFittingJob.PROCESSED_BY_LEVENBERG_MARQUARDT) == 0) {
      source.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_LEVENBERG_MARQUARDT;
      try {
        this.m_iterationCounter = new Incrementor(
            _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);
        this.m_evaluationCounter = new Incrementor(
            _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS
                * _LSSimplexFittingJob.OPTIMIZER_MAX_ITERATIONS);

        if (this.m_levenbergMarquardt == null) {
          this.m_levenbergMarquardt = new LevenbergMarquardtOptimizer();
        }

        res = this.m_levenbergMarquardt.optimize(this);
        quality = res.getRMS();

        if (MathUtils.isFinite(quality)) {
          source.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_LEAST_SQUARES;
          if (quality < source.quality) {
            dest.quality = quality;
            System.arraycopy(
                _LSSimplexFittingJob.__toArray(res.getPoint()), 0,
                dest.solution, 0, dest.solution.length);
            dest.m_processedBy = _LSSimplexFittingJob.PROCESSED_BY_LEAST_SQUARES;
            return true;
          }
        }
      } catch (@SuppressWarnings("unused") final Throwable error) {
        // ignore
      } finally {
        this.m_evaluationCounter = null;
        this.m_iterationCounter = null;
      }
    }
    return false;
  }

  /**
   * refine the current {@link #m_startVector start point} with the
   * Gauss-Newton method.
   *
   * @param source
   *          the source candidate solution
   * @param dest
   *          the destination candidate solution
   * @return {@code true} on success, {@code false} on failure
   */
  private final boolean __refineStartWithGaussNewton(
      final _Candidate source, final _Candidate dest) {
    final Optimum res;
    final double quality;

    if ((source.m_processedBy
        & _LSSimplexFittingJob.PROCESSED_BY_GAUSS_NEWTON) == 0) {
      source.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_GAUSS_NEWTON;
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

        res = this.m_gaussNewton.optimize(this);
        quality = res.getRMS();

        if (MathUtils.isFinite(quality)) {
          source.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_LEAST_SQUARES;
          if (quality < source.quality) {
            dest.quality = quality;
            System.arraycopy(
                _LSSimplexFittingJob.__toArray(res.getPoint()), 0,
                dest.solution, 0, dest.solution.length);
            dest.m_processedBy = _LSSimplexFittingJob.PROCESSED_BY_LEAST_SQUARES;
            return true;
          }
        }
      } catch (@SuppressWarnings("unused") final Throwable error) {
        // ignore
      } finally {
        this.m_evaluationCounter = null;
        this.m_iterationCounter = null;
      }
    }

    return false;
  }

  /**
   * Refine the current {@link #m_startVector start point} with a
   * least-squares method
   *
   * @param source
   *          the source candidate solution
   * @param dest
   *          the destination candidate solution
   * @return {@code true} on success, {@code false} on failure
   */
  private final boolean __refineWithLeastSquares(final _Candidate source,
      final _Candidate dest) {

    System.arraycopy(source.solution, 0, this.m_startVectorData, 0,
        this.m_startVectorData.length);
    return (this.__refineStartWithLevenbergMarquardt(source, dest) || //
        this.__refineStartWithGaussNewton(source, dest));
  }

  /**
   * refine a given solution using Nelder-Mead
   *
   * @param source
   *          the source candidate solution
   * @param dest
   *          the destination candidate solution
   * @return {@code true} on success, {@code false} on failure
   */
  @SuppressWarnings("unused")
  private final boolean __refineWithNelderMead(final _Candidate source,
      final _Candidate dest) {
    final int dim;
    PointValuePair res;
    double quality;

    if ((source.m_processedBy
        & _LSSimplexFittingJob.PROCESSED_BY_NELDER_MEAD) == 0) {
      source.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_NELDER_MEAD;
      try {
        if (this.m_simplex == null) {
          this.m_simplex = new SimplexOptimizer(1e-10d,
              Double.NEGATIVE_INFINITY);
        }
        if (this.m_objective == null) {
          this.m_objective = new ObjectiveFunction(this);
        }
        dim = source.solution.length;

        if (this.m_maxEval == null) {
          this.m_maxEval = new MaxEval(dim * dim * 300);
        }
        if (this.m_maxIter == null) {
          this.m_maxIter = new MaxIter(this.m_maxEval.getMaxEval());
        }

        res = this.m_simplex.optimize(//
            new NelderMeadSimplex(source.solution), //
            new InitialGuess(source.solution), //
            this.m_objective, this.m_maxEval, this.m_maxIter,
            GoalType.MINIMIZE);

        quality = res.getValue().doubleValue();

        if (MathUtils.isFinite(quality) && (quality < source.quality)) {
          dest.quality = quality;
          System.arraycopy(res.getPoint(), 0, dest.solution, 0,
              dest.solution.length);
          dest.m_processedBy |= _LSSimplexFittingJob.PROCESSED_BY_NELDER_MEAD;
          return true;
        }

      } catch (final Throwable error) {
        // ignore
      }
    }

    return false;
  }

  //// END: optimization routines

  /** {@inheritDoc} */
  @Override
  protected void fit() {
    final int numParameters, maxStartPointSamples;
    final _CandidateManager manager;
    final Random random;
    final double[] tempStartGuess;
    _Candidate currentSolution, nextSolution;
    IParameterGuesser guesser;
    double currentQuality;
    int index, iterations;

    // initialize and allocate all needed variables

    random = new Random();
    numParameters = this.m_function.getParameterCount();

    manager = new _CandidateManager(
        (_LSSimplexFittingJob.MAIN_LOOP_ITERATIONS * 4), numParameters);

    guesser = this.m_function.createParameterGuesser(this.m_data);

    this.m_startVectorData = tempStartGuess = new double[numParameters];
    this.m_startVector = new ArrayRealVector(tempStartGuess, false);

    maxStartPointSamples = Math.max(100,
        Math.min(10000, ((int) (Math.round(//
            2d * Math.pow(3d, numParameters))))))
        / 3;

    // Inner loop: 1) generate initial guess, 2) use least-squares
    // approach to refine, 3) use direct black-box optimizer to refine,
    // 4) if that worked, try least-squares again
    for (iterations = _LSSimplexFittingJob.MAIN_LOOP_ITERATIONS; (--iterations) >= 0;) {

      currentSolution = manager._create();

      // Find initial guess: we use the parameter guesser provided by the
      // model to create a few guesses and keep the best one
      guesser.createRandomGuess(currentSolution.solution, random);
      currentSolution.quality = this.evaluate(currentSolution.solution);

      for (index = maxStartPointSamples; (--index) >= 0;) {
        guesser.createRandomGuess(tempStartGuess, random);
        currentQuality = this.evaluate(tempStartGuess);
        if (currentQuality < currentSolution.quality) {
          // Try to get starting points which are, sort of, different.
          if (manager._isUnique(tempStartGuess,
              _LSSimplexFittingJob.MIN_REQUIRED_DISTANCE)) {
            System.arraycopy(tempStartGuess, 0, currentSolution.solution,
                0, numParameters);
            currentSolution.quality = currentQuality;
          }
        }
      }

      currentSolution = manager._tryCoalesce(currentSolution);
      nextSolution = manager._create();

      // Refine initial guess by using a least-squares solver for
      // traditional function fitting.
      if (this.__refineWithLeastSquares(currentSolution, nextSolution)) {
        currentSolution = manager._tryCoalesce(nextSolution);
        nextSolution = manager._create();
      }

      // The least-squares method may get trapped in a local optimum. We
      // need to refine its results using a direct numerical method. We
      // chose Nelder-Mead. In our tests, CMA-ES and BOBYQA did not provide
      // significantly better results in this application but take longer
      // to run.
      if (this.__refineWithNelderMead(currentSolution, nextSolution)) {
        currentSolution = manager._tryCoalesce(nextSolution);
        nextSolution = manager._create();
      }

      // If we arrived here, we have successfully applied a
      // least-squares method and refined its result with a direct,
      // black-box optimizer.
      if (this.__refineWithLeastSquares(currentSolution, nextSolution)) {
        currentSolution = manager._tryCoalesce(nextSolution);
      } else {
        manager._dispose();
      }
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
}
