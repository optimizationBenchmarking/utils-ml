package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import java.util.Random;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.fitting.leastsquares.GaussNewtonOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.util.Incrementor;
import org.optimizationBenchmarking.utils.ml.fitting.spec.FittingEvaluation;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingQualityMeasure;

/**
 * A function fitting job which has some basic provisions to utilize
 * various numerical optimization methods.
 *
 * @param <FCST>
 *          the base class for fitting candidate solutions used
 */
@SuppressWarnings("deprecation")
public abstract class OptimizationBasedFittingJob<FCST extends FittingCandidateSolution>
    extends FittingJob implements MultivariateFunction,
    LeastSquaresProblem, ConvergenceChecker<Evaluation> {

  /** Relative tolerance threshold. */
  private static final double OPTIMIZER_RELATIVE_THRESHOLD = 1e-10d;

  /** the maximum number of iterations for least squares methods */
  protected static final int DEFAULT_LEAST_SQUARES_MAX_ITERATIONS = 768;

  /** the solution was improved */
  protected static final int RET_IMPROVEMENT = 0;
  /** the application of the optimization method has failed */
  protected static final int RET_FAILED = (OptimizationBasedFittingJob.RET_IMPROVEMENT
      + 1);
  /** the solution was not improved */
  protected static final int RET_NO_IMPROVEMENT = (OptimizationBasedFittingJob.RET_FAILED
      + 1);

  /** the selected points */
  private IFittingQualityMeasure m_selected;

  /** the evaluation counter */
  private Incrementor m_evaluationCounter;
  /** the iteration counter */
  private Incrementor m_iterationCounter;

  /** the start vector */
  private ArrayRealVector m_startVector;
  /** the start vector data */
  private double[] m_startVectorData;

  /** the objective function */
  private ObjectiveFunction m_objective;
  /** the maximum evaluations */
  private MaxEval m_maxEval;
  /** the maximum iterations */
  private MaxIter m_maxIter;
  /** the shared point value pair checker */
  private __PointValuePairChecker m_pointValuePairChecker;

  /** the Gauss-Newton optimizer */
  private GaussNewtonOptimizer m_gaussNewton;
  /** the Levenberg-Marquardt optimizer */
  private LevenbergMarquardtOptimizer m_levenbergMarquardt;
  /** the simplex optimizer */
  private SimplexOptimizer m_simplex;
  /** the bobyqa optimizer */
  private BOBYQAOptimizer m_bobyqa;
  /** the CMA-ES optimizer */
  private CMAESOptimizer m_cmaes;

  /** the maximum iterations granted to least squares methods */
  private int m_leastSquaresMaxIterations;
  /** the maximum iterations granted to optimization algorithms */
  private int m_optimizerMaxIterations;

  /** the best solution found in the internal optimization steps */
  private double[] m_bestData;
  /** the best quality found in the internal optimization steps */
  private double m_bestQuality;

  /**
   * create the fitting job
   *
   * @param builder
   *          the builder
   */
  protected OptimizationBasedFittingJob(final FittingJobBuilder builder) {
    super(builder);
    final int dim;

    this.m_leastSquaresMaxIterations = OptimizationBasedFittingJob.DEFAULT_LEAST_SQUARES_MAX_ITERATIONS;

    dim = this.m_function.getParameterCount();
    this.m_optimizerMaxIterations = (dim * dim * 300);
    this.m_selected = this.m_measure;
  }

  /**
   * Set the maximum number of iterations for least squares algorithms
   *
   * @param maxIterations
   *          the maximum number of iterations for least squares algorithms
   */
  protected final void setLeastSquaresMaxIterations(
      final int maxIterations) {
    this.m_leastSquaresMaxIterations = maxIterations;
  }

  /**
   * Get the maximum number of iterations for least squares algorithms
   *
   * @return the maximum number of iterations for least squares algorithms
   */
  protected final int getLeastSquaresMaxIterations() {
    return this.m_leastSquaresMaxIterations;
  }

  /**
   * Set the maximum number of iterations for numerical optimization
   * algorithms
   *
   * @param maxIterations
   *          the maximum number of iterations for numerical optimization
   *          algorithms
   */
  protected final void setNumericalOptimizerMaxIterations(
      final int maxIterations) {
    this.m_maxEval = null;
    this.m_maxIter = null;
    this.m_optimizerMaxIterations = maxIterations;
  }

  /**
   * Get the maximum number of iterations for numerical optimization
   * algorithms
   *
   * @return the maximum number of iterations for numerical optimization
   *         algorithms
   */
  protected final int getNumericalOptimizerMaxIterations() {
    return this.m_optimizerMaxIterations;
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
    return this.m_selected.getSampleCount();
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
    vector = OptimizationBasedFittingJob.__toArray(point);

    this.m_selected.evaluate(this.m_function, vector, eval);
    if ((eval.quality < this.m_bestQuality) && (eval.quality >= 0d)) {
      this.m_bestQuality = eval.quality;
      System.arraycopy(vector, 0, this.m_bestData, 0,
          this.m_bestData.length);
      if (this.m_selected == this.m_measure) {
        this.register(eval.quality, vector);
      }
    }
    return eval;
  }

  /**
   * Choose a set of points
   *
   * @param npoints
   *          the number of points to select
   * @param random
   *          the random number generator
   */
  protected final void subselect(final int npoints, final Random random) {
    this.m_selected = this.m_measure.subselect(npoints, random);
    this.m_bestQuality = Double.POSITIVE_INFINITY;
  }

  /**
   * Select the specified points
   */
  protected final void deselectPoints() {
    this.m_selected = this.m_measure;
    this.m_bestQuality = Double.POSITIVE_INFINITY;
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double[] point) {
    final double res;

    res = this.m_selected.evaluate(this.m_function, point);
    if ((res < this.m_bestQuality) && (res >= 0d)) {
      this.m_bestQuality = res;
      System.arraycopy(point, 0, this.m_bestData, 0,
          this.m_bestData.length);
      if (this.m_selected == this.m_measure) {
        this.register(res, point);
      }
    }
    return res;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean converged(final int iteration,
      final Evaluation previous, final Evaluation current) {
    if (iteration >= this.m_leastSquaresMaxIterations) {
      return true;
    }
    return OptimizationBasedFittingJob.__check(previous.getRMS(),
        current.getRMS());
  }

  /**
   * check two double (cost) values for convergence
   *
   * @param previousValue
   *          the previous value
   * @param currentValue
   *          the current value
   * @return {@code true} on convergence, {@code false} otherwise
   */
  private static final boolean __check(final double previousValue,
      final double currentValue) {
    return (Math.abs(previousValue - currentValue) < //
    (OptimizationBasedFittingJob.OPTIMIZER_RELATIVE_THRESHOLD
        * Math.max(Math.abs(previousValue), Math.abs(currentValue))));
  }

  /**
   * Check the convergence of a point value pair.
   *
   * @param iteration
   *          the iteration
   * @param previous
   *          the previous point value pair
   * @param current
   *          the current point value pair
   * @return {@code true} on convergence, {@code false} otherwise
   */
  public final boolean converged(final int iteration,
      final PointValuePair previous, final PointValuePair current) {
    if (iteration >= this.m_optimizerMaxIterations) {
      return true;
    }
    return OptimizationBasedFittingJob.__check(
        previous.getValue().doubleValue(),
        current.getValue().doubleValue());
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
   * Copy a given array to the internal start vector.
   *
   * @param data
   *          the data array
   */
  private final void __copyToStartVector(final double[] data) {
    if (this.m_startVector == null) {
      this.m_startVectorData = data.clone();
      this.m_startVector = new ArrayRealVector(this.m_startVectorData,
          false);
    } else {
      System.arraycopy(data, 0, this.m_startVectorData, 0, data.length);
    }
  }

  /**
   * Check whether an improved fitting candidate solution can be accepted
   *
   * @param solution
   *          the solution
   * @return {@link #RET_IMPROVEMENT} by default, other codes if you want
   */
  protected int checkImprovedSolution(final FCST solution) {
    return OptimizationBasedFittingJob.RET_IMPROVEMENT;
  }

  /**
   * Produce the return value.
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  private final int __return(final FCST solution) {
    final double quality;

    quality = this.m_bestQuality;
    this.m_bestQuality = Double.POSITIVE_INFINITY;

    if ((quality < solution.quality) && (quality >= 0d)) {
      solution.assign(this.m_bestData, quality);
      return this.checkImprovedSolution(solution);
    }
    if ((quality < 0d) || (quality >= Double.POSITIVE_INFINITY)) {
      return OptimizationBasedFittingJob.RET_FAILED;
    }

    return OptimizationBasedFittingJob.RET_NO_IMPROVEMENT;
  }

  /**
   * Refine a given {@code solution} with the Levenberg-Marquardt
   * algorithm.
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithLevenbergMarquardt(final FCST solution) {
    this.m_bestQuality = Double.POSITIVE_INFINITY;

    try {
      this.m_iterationCounter = new Incrementor(
          this.m_leastSquaresMaxIterations);
      this.m_evaluationCounter = new Incrementor(
          this.m_leastSquaresMaxIterations
              * this.m_leastSquaresMaxIterations);
      this.__copyToStartVector(solution.solution);

      if (this.m_levenbergMarquardt == null) {
        this.m_levenbergMarquardt = new LevenbergMarquardtOptimizer();
      }
      this.m_levenbergMarquardt.optimize(this);
    } catch (@SuppressWarnings("unused") final Throwable error) {
      // ignored
    } finally {
      this.m_evaluationCounter = null;
      this.m_iterationCounter = null;
    }

    return this.__return(solution);
  }

  /**
   * Refine the current {@code solution} with the Gauss-Newton method.
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithGaussNewton(final FCST solution) {
    this.m_bestQuality = Double.POSITIVE_INFINITY;

    try {
      this.__copyToStartVector(solution.solution);

      this.m_iterationCounter = new Incrementor(
          this.m_leastSquaresMaxIterations);
      this.m_evaluationCounter = new Incrementor(
          this.m_leastSquaresMaxIterations
              * this.m_leastSquaresMaxIterations);

      if (this.m_gaussNewton == null) {
        this.m_gaussNewton = new GaussNewtonOptimizer(
            GaussNewtonOptimizer.Decomposition.SVD);
      }

      this.m_gaussNewton.optimize(this);
    } catch (@SuppressWarnings("unused") final Throwable error) {
      // ignored
    } finally {
      this.m_evaluationCounter = null;
      this.m_iterationCounter = null;
    }

    return this.__return(solution);
  }

  /**
   * Refine the given solution with a least-squares method
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithLeastSquares(final FCST solution) {
    final int retVal;

    switch (retVal = this.refineWithLevenbergMarquardt(solution)) {
      case RET_FAILED:
      case RET_NO_IMPROVEMENT: {
        return this.refineWithGaussNewton(solution);
      }
      default: {
        return retVal;
      }
    }
  }

  /**
   * get the maximum evaluations: large enough for Nelder-Mead and BOBYQA
   *
   * @return the maximum evaluations fitting to the maximum iterations
   */
  private final MaxEval __getMaxEval() {
    final int maxIt, numParams;
    if (this.m_maxEval == null) {
      maxIt = this.m_optimizerMaxIterations;
      numParams = this.m_function.getParameterCount();
      this.m_maxEval = new MaxEval(//
          Math.max(maxIt, Math.max(1000, //
              (maxIt * numParams * numParams * 2))));
    }
    return this.m_maxEval;
  }

  /**
   * get the maximum iterations: large enough for Nelder-Mead and BOBYQA
   *
   * @return the maximum iterations fitting to the maximum iterations
   */
  private final MaxIter __getMaxIterations() {
    if (this.m_maxIter == null) {
      this.m_maxIter = new MaxIter(this.m_optimizerMaxIterations + 1);
    }
    return this.m_maxIter;
  }

  /**
   * refine a given solution using Nelder-Mead
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithNelderMead(final FCST solution) {
    double[] bounds;
    double value;
    int index;

    this.m_bestQuality = Double.POSITIVE_INFINITY;

    try {
      if (this.m_pointValuePairChecker == null) {
        this.m_pointValuePairChecker = new __PointValuePairChecker();
      }

      if (this.m_simplex == null) {
        this.m_simplex = new SimplexOptimizer(
            this.m_pointValuePairChecker);
      }
      if (this.m_objective == null) {
        this.m_objective = new ObjectiveFunction(this);
      }

      bounds = solution.solution;
      for (index = bounds.length; (--index) >= 0;) {
        if (Math.abs(value = bounds[index]) < Double.MIN_NORMAL) {
          if (bounds == solution.solution) {
            bounds = bounds.clone();
          }
          bounds[index] = ((value < 0d) ? (-Double.MIN_NORMAL)
              : Double.MIN_NORMAL);
        }
      }

      this.m_simplex.optimize(//
          new NelderMeadSimplex(bounds), //
          new InitialGuess(solution.solution), //
          this.m_objective, this.__getMaxEval(), this.__getMaxIterations(),
          GoalType.MINIMIZE);

    } catch (@SuppressWarnings("unused") final Throwable error) {
      // unused
    }

    return this.__return(solution);
  }

  /**
   * Refine a given solution using simplex search
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithSimplexSearch(final FCST solution) {
    return this.refineWithNelderMead(solution);
  }

  /**
   * refine a given solution using BOBYQA
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithBOBYQA(final FCST solution) {
    final int dim;
    final double[] lower, upper, orig;
    double bound;
    int index;

    this.m_bestQuality = Double.POSITIVE_INFINITY;

    try {
      if (this.m_objective == null) {
        this.m_objective = new ObjectiveFunction(this);
      }
      dim = solution.solution.length;

      if (this.m_bobyqa == null) {
        this.m_bobyqa = new BOBYQAOptimizer(dim << 1);
      }

      orig = solution.solution;
      lower = new double[orig.length];
      upper = new double[orig.length];
      index = 0;
      for (final double value : orig) {
        if (value < 0d) {
          bound = (-value);
        } else {
          bound = value;
        }
        if (bound > 1e50d) {
          // try to prevent cases where BOBYQA may get problems
          return OptimizationBasedFittingJob.RET_FAILED;
        }
        if (bound < 1e-9d) {
          bound = 1e-9d;
        }
        bound *= 10d;

        lower[index] = Math.nextAfter((value - bound),
            Double.NEGATIVE_INFINITY);
        upper[index] = Math.nextUp(value + bound);
        ++index;
      }

      this.m_bobyqa.optimize(GoalType.MINIMIZE, //
          this.m_objective, //
          new InitialGuess(solution.solution), //
          new SimpleBounds(lower, upper), //
          this.__getMaxEval(), this.__getMaxIterations());

    } catch (@SuppressWarnings("unused") final Throwable error) {
      // ignored
    }

    return this.__return(solution);
  }

  /**
   * refine a given solution using CMA-ES
   *
   * @param solution
   *          the solution to refine
   * @param stddev
   *          the standard deviations
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithCMAES(final FCST solution,
      final double[] stddev) {
    final int dim;
    final double[] lower, upper, orig;
    double value, offset;
    int index;

    this.m_bestQuality = Double.POSITIVE_INFINITY;

    try {
      if (this.m_objective == null) {
        this.m_objective = new ObjectiveFunction(this);
      }

      if (this.m_pointValuePairChecker == null) {
        this.m_pointValuePairChecker = new __PointValuePairChecker();
      }

      dim = solution.solution.length;

      if (this.m_cmaes == null) {
        this.m_cmaes = new CMAESOptimizer(//
            this.m_optimizerMaxIterations, //
            0d, //
            true, //
            (this.m_optimizerMaxIterations / 10), //
            0, //
            new JDKRandomGenerator(), //
            false, //
            this.m_pointValuePairChecker);
      }

      orig = solution.solution;
      lower = new double[orig.length];
      upper = new double[orig.length];
      for (index = orig.length; (--index) >= 0;) {
        value = orig[index];
        offset = Math.max((3d * stddev[index]), 1e-10d);
        lower[index] = Math.nextAfter((value - offset),
            Double.NEGATIVE_INFINITY);
        upper[index] = Math.nextUp(value + offset);
      }

      this.m_cmaes.optimize(GoalType.MINIMIZE, //
          this.m_objective, //
          new InitialGuess(solution.solution), //
          new CMAESOptimizer.Sigma(stddev), //
          this.__getMaxEval(), this.__getMaxIterations(), //
          new CMAESOptimizer.PopulationSize(
              5 + ((int) (3 * Math.log(dim)))), //
          new SimpleBounds(lower, upper)//
      );

    } catch (@SuppressWarnings("unused") final Throwable error) {
      // ignored
    }

    return this.__return(solution);
  }

  /**
   * refine a given solution using all available means
   *
   * @param solution
   *          the solution to refine
   * @return one of the {@code RET_} codes
   */
  protected final int refineWithLeastSquaresAndSimplexSearch(
      final FCST solution) {
    boolean doLocalSearch, hasImprovement;
    int retVal, maxIterations;

    doLocalSearch = true;
    hasImprovement = false;
    loop: for (maxIterations = 100; (--maxIterations) > 0;) {
      switch (retVal = this.refineWithLeastSquares(solution)) {
        case RET_IMPROVEMENT: {
          hasImprovement = doLocalSearch = true;
          break;
        }
        case RET_FAILED:
        case RET_NO_IMPROVEMENT: {
          break;
        }
        default: {
          return retVal;
        }
      }
      if (doLocalSearch) {
        switch (this.refineWithSimplexSearch(solution)) {
          case RET_IMPROVEMENT: {
            hasImprovement = true;
            doLocalSearch = false;
            continue loop;
          }
          case RET_FAILED:
          case RET_NO_IMPROVEMENT: {
            break;
          }
          default: {
            return retVal;
          }
        }
      }
      break loop;
    }
    if (hasImprovement) {
      return OptimizationBasedFittingJob.RET_IMPROVEMENT;
    }
    return OptimizationBasedFittingJob.RET_NO_IMPROVEMENT;
  }

  //// END: optimization routines

  /** perform the fitting procedure */
  protected abstract void doFit();

  /** {@inheritDoc} */
  @Override
  protected final void fit() {
    try {
      this.m_bestData = new double[this.m_function.getParameterCount()];
      this.doFit();
    } finally {
      this.m_gaussNewton = null;
      this.m_levenbergMarquardt = null;
      this.m_cmaes = null;
      this.m_bobyqa = null;
      this.m_simplex = null;
      this.m_pointValuePairChecker = null;
      this.m_startVector = null;
      this.m_startVectorData = null;
      this.m_evaluationCounter = null;
      this.m_iterationCounter = null;
      this.m_objective = null;
      this.m_maxEval = null;
      this.m_maxIter = null;
      this.m_selected = null;
      this.m_bestData = null;
    }
  }

  /** The internal evaluation. */
  private static final class _InternalEvaluation extends FittingEvaluation
      implements Evaluation {

    /** the jacobian */
    private transient RealMatrix m_jacobian;
    /** the residuals */
    private transient RealVector m_residuals;
    /** the covariance */
    private transient RealMatrix m_covariance;
    /** the sigma */
    private transient RealVector m_sigma;
    /** the point */
    private final RealVector m_point;

    /**
     * create the internal evaluation
     *
     * @param point
     *          the point
     */
    _InternalEvaluation(final RealVector point) {
      super();
      this.m_point = point;
    }

    /** {@inheritDoc} */
    @Override
    public final RealMatrix getJacobian() {
      if (this.m_jacobian == null) {
        this.m_jacobian = new Array2DRowRealMatrix(this.jacobian, false);
      }
      return this.m_jacobian;
    }

    /** {@inheritDoc} */
    @Override
    public final RealVector getResiduals() {
      if (this.m_residuals == null) {
        this.m_residuals = new ArrayRealVector(this.residuals, false);
      }
      return this.m_residuals;
    }

    /** {@inheritDoc} */
    @Override
    public final RealVector getPoint() {
      return this.m_point;
    }

    /** {@inheritDoc} */
    @Override
    public final RealMatrix getCovariances(final double threshold) {
      final RealMatrix j, jTj;

      if (this.m_covariance == null) {

        j = this.getJacobian();
        jTj = j.transpose().multiply(j);

        this.m_covariance = new QRDecomposition(jTj, threshold).getSolver()
            .getInverse();
      }

      return this.m_covariance;
    }

    /** {@inheritDoc} */
    @Override
    public final RealVector getSigma(
        final double covarianceSingularityThreshold) {
      final RealMatrix cov;
      final RealVector sig;
      int i;

      if (this.m_sigma == null) {
        cov = this.getCovariances(covarianceSingularityThreshold);
        i = cov.getColumnDimension();
        this.m_sigma = sig = new ArrayRealVector(i);
        for (; (--i) >= 0;) {
          sig.setEntry(i, Math.sqrt(cov.getEntry(i, i)));
        }
      }
      return this.m_sigma;
    }

    /** {@inheritDoc} */
    @Override
    public final double getRMS() {
      return this.rmsError;
    }

    /** {@inheritDoc} */
    @Override
    public final double getCost() {
      return this.rsError;
    }
  }

  /** the convergence checker */
  private final class __PointValuePairChecker
      implements ConvergenceChecker<PointValuePair> {

    /** create the checker */
    __PointValuePairChecker() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final boolean converged(final int iteration,
        final PointValuePair previous, final PointValuePair current) {
      return OptimizationBasedFittingJob.this.converged(iteration,
          previous, current);
    }
  }
}
