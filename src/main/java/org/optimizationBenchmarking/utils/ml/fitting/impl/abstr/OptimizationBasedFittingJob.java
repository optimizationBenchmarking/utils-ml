package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import java.util.Random;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
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
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.util.FastMath;
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
  private __SafeBOBYQAOptimizer m_bobyqa;
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

    this.m_selected.evaluate(this.m_function, vector, true, true, eval);
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
        this.m_bobyqa = new __SafeBOBYQAOptimizer(dim << 1);
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

  /**
   * <p>
   * This is the internal drop-in replacement for
   * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
   * . The BOBYQA optimizer is quite nice, but its implementation in the
   * current commons math version (3.6.1) can sometimes go into an endless
   * loop.
   * </p>
   * <p>
   * I have reported this problem under
   * https://issues.apache.org/jira/browse/MATH-1375 and a similar problem
   * has been reported under
   * https://issues.apache.org/jira/browse/MATH-1282. What we do here is
   * basically providing a 1:1 copy of
   * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
   * , with the exception that we limit the iterations in the
   * potentially-infinite loops inside some of the methods. Once the
   * reported issues have been resolved, we will remove this implementation
   * here and use the BOBYQA optimizer directly.
   * </p>
   * <p>
   * Since this class is identical to
   * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
   * , we provide no documentation. Also, this is not to be considered as a
   * permanent solution, just as a temporary fix until the reported issues
   * are resolved.
   * </p>
   */
  private static final class __SafeBOBYQAOptimizer
      extends MultivariateOptimizer {
    /** Minimum dimension of the problem: {@value} */
    public static final int MINIMUM_PROBLEM_DIMENSION = 2;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    public static final double DEFAULT_INITIAL_RADIUS = 10.0;
    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     */
    public static final double DEFAULT_STOPPING_RADIUS = 1E-8;
    /** Constant 0. */
    private static final double ZERO = 0d;
    /** Constant 1. */
    private static final double ONE = 1d;
    /** Constant 2. */
    private static final double TWO = 2d;
    /** Constant 10. */
    private static final double TEN = 10d;
    /** Constant 16. */
    private static final double SIXTEEN = 16d;
    /** Constant 250. */
    private static final double TWO_HUNDRED_FIFTY = 250d;
    /** Constant -1. */
    private static final double MINUS_ONE = -__SafeBOBYQAOptimizer.ONE;
    /** Constant 1/2. */
    private static final double HALF = __SafeBOBYQAOptimizer.ONE / 2;
    /** Constant 1/4. */
    private static final double ONE_OVER_FOUR = __SafeBOBYQAOptimizer.ONE
        / 4;
    /** Constant 1/8. */
    private static final double ONE_OVER_EIGHT = __SafeBOBYQAOptimizer.ONE
        / 8;
    /** Constant 1/10. */
    private static final double ONE_OVER_TEN = __SafeBOBYQAOptimizer.ONE
        / 10;
    /** Constant 1/1000. */
    private static final double ONE_OVER_A_THOUSAND = __SafeBOBYQAOptimizer.ONE
        / 1000;

    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private final int m_numberOfInterpolationPoints;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private double m_initialTrustRegionRadius;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private final double m_stoppingTrustRegionRadius;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_currentBest;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private double[] m_boundDifference;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private int m_trustRegionCenterInterpolationPointIndex;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private Array2DRowRealMatrix m_bMatrix;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private Array2DRowRealMatrix m_zMatrix;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private Array2DRowRealMatrix m_interpolationPoints;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_originShift;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_fAtInterpolationPoints;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_trustRegionCenterOffset;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_gradientAtTrustRegionCenter;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_lowerDifference;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_upperDifference;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_modelSecondDerivativesParameters;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_newPoint;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_alternativeNewPoint;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_trialStepPoint;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_lagrangeValuesAtNewPoint;
    /** {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer} */
    private ArrayRealVector m_modelSecondDerivativesValues;

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param numberOfInterpolationPoints
     *          Number of interpolation conditions. For a problem of
     *          dimension {@code n}, its value must be in the interval
     *          {@code [n+2, (n+1)(n+2)/2]}. Choices that exceed
     *          {@code 2n+1} are not recommended.
     */
    __SafeBOBYQAOptimizer(final int numberOfInterpolationPoints) {
      super(null); // No custom convergence criterion.
      this.m_numberOfInterpolationPoints = numberOfInterpolationPoints;
      this.m_initialTrustRegionRadius = __SafeBOBYQAOptimizer.DEFAULT_INITIAL_RADIUS;
      this.m_stoppingTrustRegionRadius = __SafeBOBYQAOptimizer.DEFAULT_STOPPING_RADIUS;
    }

    /** {@inheritDoc} */
    @Override
    protected final PointValuePair doOptimize() {
      final double[] lowerBound = this.getLowerBound();
      final double[] upperBound = this.getUpperBound();

      this.__setup(lowerBound, upperBound);

      this.m_currentBest = new ArrayRealVector(this.getStartPoint());

      final double value = this.__bobyqa(lowerBound, upperBound);

      return new PointValuePair(this.m_currentBest.getDataRef(), value);
    }

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param lowerBound
     *          Lower bounds.
     * @param upperBound
     *          Upper bounds.
     * @return the value of the objective at the optimum or
     *         {@link Double#POSITIVE_INFINITY} if we got stuck and had to
     *         abort a potential infinite loop.
     */
    private double __bobyqa(final double[] lowerBound,
        final double[] upperBound) {

      final int n = this.m_currentBest.getDimension();

      for (int j = 0; j < n; j++) {
        final double boundDiff = this.m_boundDifference[j];
        this.m_lowerDifference.setEntry(j,
            lowerBound[j] - this.m_currentBest.getEntry(j));
        this.m_upperDifference.setEntry(j,
            upperBound[j] - this.m_currentBest.getEntry(j));
        if (this.m_lowerDifference
            .getEntry(j) >= -this.m_initialTrustRegionRadius) {
          if (this.m_lowerDifference
              .getEntry(j) >= __SafeBOBYQAOptimizer.ZERO) {
            this.m_currentBest.setEntry(j, lowerBound[j]);
            this.m_lowerDifference.setEntry(j, __SafeBOBYQAOptimizer.ZERO);
            this.m_upperDifference.setEntry(j, boundDiff);
          } else {
            this.m_currentBest.setEntry(j,
                lowerBound[j] + this.m_initialTrustRegionRadius);
            this.m_lowerDifference.setEntry(j,
                -this.m_initialTrustRegionRadius);
            // Computing MAX
            final double deltaOne = upperBound[j]
                - this.m_currentBest.getEntry(j);
            this.m_upperDifference.setEntry(j,
                FastMath.max(deltaOne, this.m_initialTrustRegionRadius));
          }
        } else
          if (this.m_upperDifference
              .getEntry(j) <= this.m_initialTrustRegionRadius) {
            if (this.m_upperDifference
                .getEntry(j) <= __SafeBOBYQAOptimizer.ZERO) {
              this.m_currentBest.setEntry(j, upperBound[j]);
              this.m_lowerDifference.setEntry(j, -boundDiff);
              this.m_upperDifference.setEntry(j,
                  __SafeBOBYQAOptimizer.ZERO);
            } else {
              this.m_currentBest.setEntry(j,
                  upperBound[j] - this.m_initialTrustRegionRadius);
              // Computing MIN
              final double deltaOne = lowerBound[j]
                  - this.m_currentBest.getEntry(j);
              final double deltaTwo = -this.m_initialTrustRegionRadius;
              this.m_lowerDifference.setEntry(j,
                  FastMath.min(deltaOne, deltaTwo));
              this.m_upperDifference.setEntry(j,
                  this.m_initialTrustRegionRadius);
            }
          }
      }

      return this.__bobyqb(lowerBound, upperBound);
    }

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param lowerBound
     *          Lower bounds.
     * @param upperBound
     *          Upper bounds.
     * @return the value of the objective at the optimum, or
     *         {@link Double#POSITIVE_INFINITY} if we got stuck and had to
     *         abort a potential infinite loop.
     */
    @SuppressWarnings("fallthrough")
    private double __bobyqb(final double[] lowerBound,
        final double[] upperBound) {

      final int n = this.m_currentBest.getDimension();
      final int npt = this.m_numberOfInterpolationPoints;
      final int np = n + 1;
      final int nptm = npt - np;
      final int nh = (n * np) / 2;

      final ArrayRealVector work1 = new ArrayRealVector(n);
      final ArrayRealVector work2 = new ArrayRealVector(npt);
      final ArrayRealVector work3 = new ArrayRealVector(npt);

      double cauchy = Double.NaN;
      double alpha = Double.NaN;
      double dsq = Double.NaN;
      double crvmin = Double.NaN;

      // Set some constants.
      // Parameter adjustments

      // Function Body

      // The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT,
      // HQ, PQ,
      // BMAT and ZMAT for the first iteration, with the corresponding
      // values of
      // of NF and KOPT, which are the number of calls of CALFUN so far and
      // the
      // index of the interpolation point at the trust region centre. Then
      // the
      // initial XOPT is set too. The branch to label 720 occurs if MAXFUN
      // is
      // less than NPT. GOPT will be updated if KOPT is different from
      // KBASE.

      this.m_trustRegionCenterInterpolationPointIndex = 0;

      this.__prelim(lowerBound, upperBound);
      double xoptsq = __SafeBOBYQAOptimizer.ZERO;
      for (int i = 0; i < n; i++) {
        this.m_trustRegionCenterOffset.setEntry(i,
            this.m_interpolationPoints.getEntry(
                this.m_trustRegionCenterInterpolationPointIndex, i));
        // Computing 2nd power
        final double deltaOne = this.m_trustRegionCenterOffset.getEntry(i);
        xoptsq += deltaOne * deltaOne;
      }
      double fsave = this.m_fAtInterpolationPoints.getEntry(0);
      final int kbase = 0;

      // Complete the settings that are required for the iterative
      // procedure.

      int ntrits = 0;
      int itest = 0;
      int knew = 0;
      int nfsav = this.getEvaluations();
      double rho = this.m_initialTrustRegionRadius;
      double delta = rho;
      double diffa = __SafeBOBYQAOptimizer.ZERO;
      double diffb = __SafeBOBYQAOptimizer.ZERO;
      double diffc = __SafeBOBYQAOptimizer.ZERO;
      double f = __SafeBOBYQAOptimizer.ZERO;
      double beta = __SafeBOBYQAOptimizer.ZERO;
      double adelt = __SafeBOBYQAOptimizer.ZERO;
      double denom = __SafeBOBYQAOptimizer.ZERO;
      double ratio = __SafeBOBYQAOptimizer.ZERO;
      double dnorm = __SafeBOBYQAOptimizer.ZERO;
      double scaden = __SafeBOBYQAOptimizer.ZERO;
      double biglsq = __SafeBOBYQAOptimizer.ZERO;
      double distsq = __SafeBOBYQAOptimizer.ZERO;

      // Update GOPT if necessary before the first iteration and after each
      // call of RESCUE that makes a call of CALFUN.

      int state = 20;
      for (int indexer = 100000; (--indexer) >= 0;) {
        switch (state) {
          case 20: {

            if (this.m_trustRegionCenterInterpolationPointIndex != kbase) {
              int ih = 0;
              for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                  if (i < j) {
                    this.m_gradientAtTrustRegionCenter.setEntry(j,
                        this.m_gradientAtTrustRegionCenter.getEntry(j)
                            + (this.m_modelSecondDerivativesValues
                                .getEntry(ih)
                                * this.m_trustRegionCenterOffset
                                    .getEntry(i)));
                  }
                  this.m_gradientAtTrustRegionCenter.setEntry(i,
                      this.m_gradientAtTrustRegionCenter.getEntry(i)
                          + (this.m_modelSecondDerivativesValues
                              .getEntry(ih)
                              * this.m_trustRegionCenterOffset
                                  .getEntry(j)));
                  ih++;
                }
              }
              if (this.getEvaluations() > npt) {
                for (int k = 0; k < npt; k++) {
                  double temp = __SafeBOBYQAOptimizer.ZERO;
                  for (int j = 0; j < n; j++) {
                    temp += this.m_interpolationPoints.getEntry(k, j)
                        * this.m_trustRegionCenterOffset.getEntry(j);
                  }
                  temp *= this.m_modelSecondDerivativesParameters
                      .getEntry(k);
                  for (int i = 0; i < n; i++) {
                    this.m_gradientAtTrustRegionCenter.setEntry(i,
                        this.m_gradientAtTrustRegionCenter.getEntry(i)
                            + (temp * this.m_interpolationPoints
                                .getEntry(k, i)));
                  }
                }
              }
            }

            // Generate the next point in the trust region that provides a
            // small value
            // of the quadratic model subject to the constraints on the
            // variables.
            // The int NTRITS is set to the number "trust region"
            // iterations that
            // have occurred since the last "alternative" iteration. If the
            // length
            // of XNEW-XOPT is less than HALF*RHO, however, then there is a
            // branch to
            // label 650 or 680 with NTRITS=-1, instead of calculating F at
            // XNEW.

          }
          case 60: {

            final ArrayRealVector gnew = new ArrayRealVector(n);
            final ArrayRealVector xbdi = new ArrayRealVector(n);
            final ArrayRealVector s = new ArrayRealVector(n);
            final ArrayRealVector hs = new ArrayRealVector(n);
            final ArrayRealVector hred = new ArrayRealVector(n);

            final double[] dsqCrvmin = this.__trsbox(delta, gnew, xbdi, s,
                hs, hred);
            if (dsqCrvmin == null) {
              return Double.POSITIVE_INFINITY;
            }
            dsq = dsqCrvmin[0];
            crvmin = dsqCrvmin[1];

            // Computing MIN
            double deltaOne = delta;
            final double deltaTwo = FastMath.sqrt(dsq);
            dnorm = FastMath.min(deltaOne, deltaTwo);
            if (dnorm < (__SafeBOBYQAOptimizer.HALF * rho)) {
              ntrits = -1;
              // Computing 2nd power
              deltaOne = __SafeBOBYQAOptimizer.TEN * rho;
              distsq = deltaOne * deltaOne;
              if (this.getEvaluations() <= (nfsav + 2)) {
                state = 650;
                break;
              }

              // The following choice between labels 650 and 680 depends on
              // whether or
              // not our work with the current RHO seems to be complete.
              // Either RHO is
              // decreased or termination occurs if the errors in the
              // quadratic model at
              // the last three interpolation points compare favourably
              // with predictions
              // of likely improvements to the model within distance
              // HALF*RHO of XOPT.

              // Computing MAX
              deltaOne = FastMath.max(diffa, diffb);
              final double errbig = FastMath.max(deltaOne, diffc);
              final double frhosq = rho
                  * __SafeBOBYQAOptimizer.ONE_OVER_EIGHT * rho;
              if ((crvmin > __SafeBOBYQAOptimizer.ZERO)
                  && (errbig > (frhosq * crvmin))) {
                state = 650;
                break;
              }
              final double bdtol = errbig / rho;
              for (int j = 0; j < n; j++) {
                double bdtest = bdtol;
                if (this.m_newPoint.getEntry(j) == this.m_lowerDifference
                    .getEntry(j)) {
                  bdtest = work1.getEntry(j);
                }
                if (this.m_newPoint.getEntry(j) == this.m_upperDifference
                    .getEntry(j)) {
                  bdtest = -work1.getEntry(j);
                }
                if (bdtest < bdtol) {
                  double curv = this.m_modelSecondDerivativesValues
                      .getEntry((j + (j * j)) / 2);
                  for (int k = 0; k < npt; k++) {
                    // Computing 2nd power
                    final double d1 = this.m_interpolationPoints
                        .getEntry(k, j);
                    curv += this.m_modelSecondDerivativesParameters
                        .getEntry(k) * (d1 * d1);
                  }
                  bdtest += __SafeBOBYQAOptimizer.HALF * curv * rho;
                  if (bdtest < bdtol) {
                    state = 650;
                    break;
                  }
                }
              }
              state = 680;
              break;
            }
            ++ntrits;

            // Severe cancellation is likely to occur if XOPT is too far
            // from XBASE.
            // If the following test holds, then XBASE is shifted so that
            // XOPT becomes
            // zero. The appropriate changes are made to BMAT and to the
            // second
            // derivatives of the current model, beginning with the changes
            // to BMAT
            // that do not depend on ZMAT. VLAG is used temporarily for
            // working space.

          }
          case 90: {

            if (dsq <= (xoptsq
                * __SafeBOBYQAOptimizer.ONE_OVER_A_THOUSAND)) {
              final double fracsq = xoptsq
                  * __SafeBOBYQAOptimizer.ONE_OVER_FOUR;
              double sumpq = __SafeBOBYQAOptimizer.ZERO;
              // final RealVector sumVector
              // = new ArrayRealVector(npt, -HALF *
              // xoptsq).add(m_interpolationPoints.operate(trustRegionCenter));
              for (int k = 0; k < npt; k++) {
                sumpq += this.m_modelSecondDerivativesParameters
                    .getEntry(k);
                double sum = -__SafeBOBYQAOptimizer.HALF * xoptsq;
                for (int i = 0; i < n; i++) {
                  sum += this.m_interpolationPoints.getEntry(k, i)
                      * this.m_trustRegionCenterOffset.getEntry(i);
                }
                work2.setEntry(k, sum);
                final double temp = fracsq
                    - (__SafeBOBYQAOptimizer.HALF * sum);
                for (int i = 0; i < n; i++) {
                  work1.setEntry(i, this.m_bMatrix.getEntry(k, i));
                  this.m_lagrangeValuesAtNewPoint.setEntry(i, (sum
                      * this.m_interpolationPoints.getEntry(k, i))
                      + (temp
                          * this.m_trustRegionCenterOffset.getEntry(i)));
                  final int ip = npt + i;
                  for (int j = 0; j <= i; j++) {
                    this.m_bMatrix.setEntry(ip, j,
                        this.m_bMatrix.getEntry(ip, j)
                            + (work1.getEntry(i)
                                * this.m_lagrangeValuesAtNewPoint
                                    .getEntry(j))
                        + (this.m_lagrangeValuesAtNewPoint.getEntry(i)
                            * work1.getEntry(j)));
                  }
                }
              }

              // Then the revisions of BMAT that depend on ZMAT are
              // calculated.

              for (int m = 0; m < nptm; m++) {
                double sumz = __SafeBOBYQAOptimizer.ZERO;
                double sumw = __SafeBOBYQAOptimizer.ZERO;
                for (int k = 0; k < npt; k++) {
                  sumz += this.m_zMatrix.getEntry(k, m);
                  this.m_lagrangeValuesAtNewPoint.setEntry(k,
                      work2.getEntry(k) * this.m_zMatrix.getEntry(k, m));
                  sumw += this.m_lagrangeValuesAtNewPoint.getEntry(k);
                }
                for (int j = 0; j < n; j++) {
                  double sum = ((fracsq * sumz)
                      - (__SafeBOBYQAOptimizer.HALF * sumw))
                      * this.m_trustRegionCenterOffset.getEntry(j);
                  for (int k = 0; k < npt; k++) {
                    sum += this.m_lagrangeValuesAtNewPoint.getEntry(k)
                        * this.m_interpolationPoints.getEntry(k, j);
                  }
                  work1.setEntry(j, sum);
                  for (int k = 0; k < npt; k++) {
                    this.m_bMatrix.setEntry(k, j,
                        this.m_bMatrix.getEntry(k, j)
                            + (sum * this.m_zMatrix.getEntry(k, m)));
                  }
                }
                for (int i = 0; i < n; i++) {
                  final int ip = i + npt;
                  final double temp = work1.getEntry(i);
                  for (int j = 0; j <= i; j++) {
                    this.m_bMatrix.setEntry(ip, j,
                        this.m_bMatrix.getEntry(ip, j)
                            + (temp * work1.getEntry(j)));
                  }
                }
              }

              // The following instructions complete the shift, including
              // the changes
              // to the second derivative parameters of the quadratic
              // model.

              int ih = 0;
              for (int j = 0; j < n; j++) {
                work1.setEntry(j, -__SafeBOBYQAOptimizer.HALF * sumpq
                    * this.m_trustRegionCenterOffset.getEntry(j));
                for (int k = 0; k < npt; k++) {
                  work1.setEntry(j, work1.getEntry(j)
                      + (this.m_modelSecondDerivativesParameters.getEntry(
                          k) * this.m_interpolationPoints.getEntry(k, j)));
                  this.m_interpolationPoints.setEntry(k, j,
                      this.m_interpolationPoints.getEntry(k, j)
                          - this.m_trustRegionCenterOffset.getEntry(j));
                }
                for (int i = 0; i <= j; i++) {
                  this.m_modelSecondDerivativesValues.setEntry(ih,
                      this.m_modelSecondDerivativesValues.getEntry(ih)
                          + (work1.getEntry(i)
                              * this.m_trustRegionCenterOffset.getEntry(j))
                          + (this.m_trustRegionCenterOffset.getEntry(i)
                              * work1.getEntry(j)));
                  this.m_bMatrix.setEntry(npt + i, j,
                      this.m_bMatrix.getEntry(npt + j, i));
                  ih++;
                }
              }
              for (int i = 0; i < n; i++) {
                this.m_originShift.setEntry(i,
                    this.m_originShift.getEntry(i)
                        + this.m_trustRegionCenterOffset.getEntry(i));
                this.m_newPoint.setEntry(i, this.m_newPoint.getEntry(i)
                    - this.m_trustRegionCenterOffset.getEntry(i));
                this.m_lowerDifference.setEntry(i,
                    this.m_lowerDifference.getEntry(i)
                        - this.m_trustRegionCenterOffset.getEntry(i));
                this.m_upperDifference.setEntry(i,
                    this.m_upperDifference.getEntry(i)
                        - this.m_trustRegionCenterOffset.getEntry(i));
                this.m_trustRegionCenterOffset.setEntry(i,
                    __SafeBOBYQAOptimizer.ZERO);
              }
              xoptsq = __SafeBOBYQAOptimizer.ZERO;
            }
            if (ntrits == 0) {
              state = 210;
              break;
            }
            state = 230;
            break;

            // XBASE is also moved to XOPT by a call of RESCUE. This
            // calculation is
            // more expensive than the previous shift, because new matrices
            // BMAT and
            // ZMAT are generated from scratch, which may include the
            // replacement of
            // interpolation points whose positions seem to be causing near
            // linear
            // dependence in the interpolation conditions. Therefore RESCUE
            // is called
            // only if rounding errors have reduced by at least a factor of
            // two the
            // denominator of the formula for updating the H matrix. It
            // provides a
            // useful safeguard, but is not invoked in most applications of
            // BOBYQA.

          }
          case 210: {

            // Pick two alternative vectors of variables, relative to
            // XBASE, that
            // are suitable as new positions of the KNEW-th interpolation
            // point.
            // Firstly, XNEW is set to the point on a line through XOPT and
            // another
            // interpolation point that minimizes the predicted value of
            // the next
            // denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to
            // the SL
            // and SU bounds. Secondly, XALT is set to the best feasible
            // point on
            // a constrained version of the Cauchy step of the KNEW-th
            // Lagrange
            // function, the corresponding value of the square of this
            // function
            // being returned in CAUCHY. The choice between these
            // alternatives is
            // going to be made when the denominator is calculated.

            final double[] alphaCauchy = this.__altmov(knew, adelt);
            alpha = alphaCauchy[0];
            cauchy = alphaCauchy[1];

            for (int i = 0; i < n; i++) {
              this.m_trialStepPoint.setEntry(i, this.m_newPoint.getEntry(i)
                  - this.m_trustRegionCenterOffset.getEntry(i));
            }

            // Calculate VLAG and BETA for the current choice of D. The
            // scalar
            // product of D with XPT(K,.) is going to be held in W(NPT+K)
            // for
            // use when VQUAD is calculated.

          }
          case 230: {

            for (int k = 0; k < npt; k++) {
              double suma = __SafeBOBYQAOptimizer.ZERO;
              double sumb = __SafeBOBYQAOptimizer.ZERO;
              double sum = __SafeBOBYQAOptimizer.ZERO;
              for (int j = 0; j < n; j++) {
                suma += this.m_interpolationPoints.getEntry(k, j)
                    * this.m_trialStepPoint.getEntry(j);
                sumb += this.m_interpolationPoints.getEntry(k, j)
                    * this.m_trustRegionCenterOffset.getEntry(j);
                sum += this.m_bMatrix.getEntry(k, j)
                    * this.m_trialStepPoint.getEntry(j);
              }
              work3.setEntry(k,
                  suma * ((__SafeBOBYQAOptimizer.HALF * suma) + sumb));
              this.m_lagrangeValuesAtNewPoint.setEntry(k, sum);
              work2.setEntry(k, suma);
            }
            beta = __SafeBOBYQAOptimizer.ZERO;
            for (int m = 0; m < nptm; m++) {
              double sum = __SafeBOBYQAOptimizer.ZERO;
              for (int k = 0; k < npt; k++) {
                sum += this.m_zMatrix.getEntry(k, m) * work3.getEntry(k);
              }
              beta -= sum * sum;
              for (int k = 0; k < npt; k++) {
                this.m_lagrangeValuesAtNewPoint.setEntry(k,
                    this.m_lagrangeValuesAtNewPoint.getEntry(k)
                        + (sum * this.m_zMatrix.getEntry(k, m)));
              }
            }
            dsq = __SafeBOBYQAOptimizer.ZERO;
            double bsum = __SafeBOBYQAOptimizer.ZERO;
            double dx = __SafeBOBYQAOptimizer.ZERO;
            for (int j = 0; j < n; j++) {
              // Computing 2nd power
              final double d1 = this.m_trialStepPoint.getEntry(j);
              dsq += d1 * d1;
              double sum = __SafeBOBYQAOptimizer.ZERO;
              for (int k = 0; k < npt; k++) {
                sum += work3.getEntry(k) * this.m_bMatrix.getEntry(k, j);
              }
              bsum += sum * this.m_trialStepPoint.getEntry(j);
              final int jp = npt + j;
              for (int i = 0; i < n; i++) {
                sum += this.m_bMatrix.getEntry(jp, i)
                    * this.m_trialStepPoint.getEntry(i);
              }
              this.m_lagrangeValuesAtNewPoint.setEntry(jp, sum);
              bsum += sum * this.m_trialStepPoint.getEntry(j);
              dx += this.m_trialStepPoint.getEntry(j)
                  * this.m_trustRegionCenterOffset.getEntry(j);
            }

            beta = ((dx * dx) + (dsq
                * (xoptsq + dx + dx + (__SafeBOBYQAOptimizer.HALF * dsq)))
                + beta) - bsum; // Original

            this.m_lagrangeValuesAtNewPoint.setEntry(
                this.m_trustRegionCenterInterpolationPointIndex,
                this.m_lagrangeValuesAtNewPoint.getEntry(
                    this.m_trustRegionCenterInterpolationPointIndex)
                    + __SafeBOBYQAOptimizer.ONE);

            // If NTRITS is zero, the denominator may be increased by
            // replacing
            // the step D of ALTMOV by a Cauchy step. Then RESCUE may be
            // called if
            // rounding errors have damaged the chosen denominator.

            if (ntrits == 0) {
              // Computing 2nd power
              final double d1 = this.m_lagrangeValuesAtNewPoint
                  .getEntry(knew);
              denom = (d1 * d1) + (alpha * beta);
              if ((denom < cauchy)
                  && (cauchy > __SafeBOBYQAOptimizer.ZERO)) {
                for (int i = 0; i < n; i++) {
                  this.m_newPoint.setEntry(i,
                      this.m_alternativeNewPoint.getEntry(i));
                  this.m_trialStepPoint.setEntry(i,
                      this.m_newPoint.getEntry(i)
                          - this.m_trustRegionCenterOffset.getEntry(i));
                }
                cauchy = __SafeBOBYQAOptimizer.ZERO;

                state = 230;
                break;
              }
              // Alternatively, if NTRITS is positive, then set KNEW to the
              // index of
              // the next interpolation point to be deleted to make room
              // for a trust
              // region step. Again RESCUE may be called if rounding errors
              // have damaged_
              // the chosen denominator, which is the reason for attempting
              // to select
              // KNEW before calculating the next value of the objective
              // function.

            } else {
              final double delsq = delta * delta;
              scaden = __SafeBOBYQAOptimizer.ZERO;
              biglsq = __SafeBOBYQAOptimizer.ZERO;
              knew = 0;
              for (int k = 0; k < npt; k++) {
                if (k == this.m_trustRegionCenterInterpolationPointIndex) {
                  continue;
                }
                double hdiag = __SafeBOBYQAOptimizer.ZERO;
                for (int m = 0; m < nptm; m++) {
                  // Computing 2nd power
                  final double d1 = this.m_zMatrix.getEntry(k, m);
                  hdiag += d1 * d1;
                }
                // Computing 2nd power
                final double d2 = this.m_lagrangeValuesAtNewPoint
                    .getEntry(k);
                final double den = (beta * hdiag) + (d2 * d2);
                distsq = __SafeBOBYQAOptimizer.ZERO;
                for (int j = 0; j < n; j++) {
                  // Computing 2nd power
                  final double d3 = this.m_interpolationPoints.getEntry(k,
                      j) - this.m_trustRegionCenterOffset.getEntry(j);
                  distsq += d3 * d3;
                }
                // Computing MAX
                // Computing 2nd power
                final double d4 = distsq / delsq;
                final double temp = FastMath.max(__SafeBOBYQAOptimizer.ONE,
                    d4 * d4);
                if ((temp * den) > scaden) {
                  scaden = temp * den;
                  knew = k;
                  denom = den;
                }
                // Computing MAX
                // Computing 2nd power
                final double d5 = this.m_lagrangeValuesAtNewPoint
                    .getEntry(k);
                biglsq = FastMath.max(biglsq, temp * (d5 * d5));
              }
            }

            // Put the variables for the next calculation of the objective
            // function
            // in XNEW, with any adjustments for the bounds.

            // Calculate the value of the objective function at XBASE+XNEW,
            // unless
            // the limit on the number of calculations of F has been
            // reached.

          }
          case 360: {

            for (int i = 0; i < n; i++) {
              // Computing MIN
              // Computing MAX
              final double d3 = lowerBound[i];
              final double d4 = this.m_originShift.getEntry(i)
                  + this.m_newPoint.getEntry(i);
              final double d1 = FastMath.max(d3, d4);
              final double d2 = upperBound[i];
              this.m_currentBest.setEntry(i, FastMath.min(d1, d2));
              if (this.m_newPoint.getEntry(i) == this.m_lowerDifference
                  .getEntry(i)) {
                this.m_currentBest.setEntry(i, lowerBound[i]);
              }
              if (this.m_newPoint.getEntry(i) == this.m_upperDifference
                  .getEntry(i)) {
                this.m_currentBest.setEntry(i, upperBound[i]);
              }
            }

            f = this.computeObjectiveValue(this.m_currentBest.toArray());

            if (ntrits == -1) {
              fsave = f;
              state = 720;
              break;
            }

            // Use the quadratic model to predict the change in F due to
            // the step D,
            // and set DIFF to the error of this prediction.

            final double fopt = this.m_fAtInterpolationPoints
                .getEntry(this.m_trustRegionCenterInterpolationPointIndex);
            double vquad = __SafeBOBYQAOptimizer.ZERO;
            int ih = 0;
            for (int j = 0; j < n; j++) {
              vquad += this.m_trialStepPoint.getEntry(j)
                  * this.m_gradientAtTrustRegionCenter.getEntry(j);
              for (int i = 0; i <= j; i++) {
                double temp = this.m_trialStepPoint.getEntry(i)
                    * this.m_trialStepPoint.getEntry(j);
                if (i == j) {
                  temp *= __SafeBOBYQAOptimizer.HALF;
                }
                vquad += this.m_modelSecondDerivativesValues.getEntry(ih)
                    * temp;
                ih++;
              }
            }
            for (int k = 0; k < npt; k++) {
              // Computing 2nd power
              final double d1 = work2.getEntry(k);
              final double d2 = d1 * d1; // "d1" must be squared first to
                                         // prevent test failures.
              vquad += __SafeBOBYQAOptimizer.HALF
                  * this.m_modelSecondDerivativesParameters.getEntry(k)
                  * d2;
            }
            final double diff = f - fopt - vquad;
            diffc = diffb;
            diffb = diffa;
            diffa = FastMath.abs(diff);
            if (dnorm > rho) {
              nfsav = this.getEvaluations();
            }

            // Pick the next value of DELTA after a trust region step.

            if (ntrits > 0) {
              if (vquad >= __SafeBOBYQAOptimizer.ZERO) {
                throw new MathIllegalStateException(
                    LocalizedFormats.TRUST_REGION_STEP_FAILED,
                    Double.valueOf(vquad));
              }
              ratio = (f - fopt) / vquad;
              final double hDelta = __SafeBOBYQAOptimizer.HALF * delta;
              if (ratio <= __SafeBOBYQAOptimizer.ONE_OVER_TEN) {
                // Computing MIN
                delta = FastMath.min(hDelta, dnorm);
              } else
                if (ratio <= .7) {
                  // Computing MAX
                  delta = FastMath.max(hDelta, dnorm);
                } else {
                  // Computing MAX
                  delta = FastMath.max(hDelta, 2 * dnorm);
                }
              if (delta <= (rho * 1.5)) {
                delta = rho;
              }

              // Recalculate KNEW and DENOM if the new F is less than FOPT.

              if (f < fopt) {
                final int ksav = knew;
                final double densav = denom;
                final double delsq = delta * delta;
                scaden = __SafeBOBYQAOptimizer.ZERO;
                biglsq = __SafeBOBYQAOptimizer.ZERO;
                knew = 0;
                for (int k = 0; k < npt; k++) {
                  double hdiag = __SafeBOBYQAOptimizer.ZERO;
                  for (int m = 0; m < nptm; m++) {
                    // Computing 2nd power
                    final double d1 = this.m_zMatrix.getEntry(k, m);
                    hdiag += d1 * d1;
                  }
                  // Computing 2nd power
                  final double d1 = this.m_lagrangeValuesAtNewPoint
                      .getEntry(k);
                  final double den = (beta * hdiag) + (d1 * d1);
                  distsq = __SafeBOBYQAOptimizer.ZERO;
                  for (int j = 0; j < n; j++) {
                    // Computing 2nd power
                    final double d2 = this.m_interpolationPoints
                        .getEntry(k, j) - this.m_newPoint.getEntry(j);
                    distsq += d2 * d2;
                  }
                  // Computing MAX
                  // Computing 2nd power
                  final double d3 = distsq / delsq;
                  final double temp = FastMath
                      .max(__SafeBOBYQAOptimizer.ONE, d3 * d3);
                  if ((temp * den) > scaden) {
                    scaden = temp * den;
                    knew = k;
                    denom = den;
                  }
                  // Computing MAX
                  // Computing 2nd power
                  final double d4 = this.m_lagrangeValuesAtNewPoint
                      .getEntry(k);
                  final double d5 = temp * (d4 * d4);
                  biglsq = FastMath.max(biglsq, d5);
                }
                if (scaden <= (__SafeBOBYQAOptimizer.HALF * biglsq)) {
                  knew = ksav;
                  denom = densav;
                }
              }
            }

            // Update BMAT and ZMAT, so that the KNEW-th interpolation
            // point can be
            // moved. Also update the second derivative terms of the model.

            this.__update(beta, denom, knew);

            ih = 0;
            final double pqold = this.m_modelSecondDerivativesParameters
                .getEntry(knew);
            this.m_modelSecondDerivativesParameters.setEntry(knew,
                __SafeBOBYQAOptimizer.ZERO);
            for (int i = 0; i < n; i++) {
              final double temp = pqold
                  * this.m_interpolationPoints.getEntry(knew, i);
              for (int j = 0; j <= i; j++) {
                this.m_modelSecondDerivativesValues.setEntry(ih,
                    this.m_modelSecondDerivativesValues.getEntry(ih)
                        + (temp * this.m_interpolationPoints.getEntry(knew,
                            j)));
                ih++;
              }
            }
            for (int m = 0; m < nptm; m++) {
              final double temp = diff * this.m_zMatrix.getEntry(knew, m);
              for (int k = 0; k < npt; k++) {
                this.m_modelSecondDerivativesParameters.setEntry(k,
                    this.m_modelSecondDerivativesParameters.getEntry(k)
                        + (temp * this.m_zMatrix.getEntry(k, m)));
              }
            }

            // Include the new interpolation point, and make the changes to
            // GOPT at
            // the old XOPT that are caused by the updating of the
            // quadratic model.

            this.m_fAtInterpolationPoints.setEntry(knew, f);
            for (int i = 0; i < n; i++) {
              this.m_interpolationPoints.setEntry(knew, i,
                  this.m_newPoint.getEntry(i));
              work1.setEntry(i, this.m_bMatrix.getEntry(knew, i));
            }
            for (int k = 0; k < npt; k++) {
              double suma = __SafeBOBYQAOptimizer.ZERO;
              for (int m = 0; m < nptm; m++) {
                suma += this.m_zMatrix.getEntry(knew, m)
                    * this.m_zMatrix.getEntry(k, m);
              }
              double sumb = __SafeBOBYQAOptimizer.ZERO;
              for (int j = 0; j < n; j++) {
                sumb += this.m_interpolationPoints.getEntry(k, j)
                    * this.m_trustRegionCenterOffset.getEntry(j);
              }
              final double temp = suma * sumb;
              for (int i = 0; i < n; i++) {
                work1.setEntry(i, work1.getEntry(i)
                    + (temp * this.m_interpolationPoints.getEntry(k, i)));
              }
            }
            for (int i = 0; i < n; i++) {
              this.m_gradientAtTrustRegionCenter.setEntry(i,
                  this.m_gradientAtTrustRegionCenter.getEntry(i)
                      + (diff * work1.getEntry(i)));
            }

            // Update XOPT, GOPT and KOPT if the new calculated F is less
            // than FOPT.

            if (f < fopt) {
              this.m_trustRegionCenterInterpolationPointIndex = knew;
              xoptsq = __SafeBOBYQAOptimizer.ZERO;
              ih = 0;
              for (int j = 0; j < n; j++) {
                this.m_trustRegionCenterOffset.setEntry(j,
                    this.m_newPoint.getEntry(j));
                // Computing 2nd power
                final double d1 = this.m_trustRegionCenterOffset
                    .getEntry(j);
                xoptsq += d1 * d1;
                for (int i = 0; i <= j; i++) {
                  if (i < j) {
                    this.m_gradientAtTrustRegionCenter.setEntry(j,
                        this.m_gradientAtTrustRegionCenter.getEntry(j)
                            + (this.m_modelSecondDerivativesValues
                                .getEntry(ih)
                                * this.m_trialStepPoint.getEntry(i)));
                  }
                  this.m_gradientAtTrustRegionCenter.setEntry(i,
                      this.m_gradientAtTrustRegionCenter.getEntry(i)
                          + (this.m_modelSecondDerivativesValues.getEntry(
                              ih) * this.m_trialStepPoint.getEntry(j)));
                  ih++;
                }
              }
              for (int k = 0; k < npt; k++) {
                double temp = __SafeBOBYQAOptimizer.ZERO;
                for (int j = 0; j < n; j++) {
                  temp += this.m_interpolationPoints.getEntry(k, j)
                      * this.m_trialStepPoint.getEntry(j);
                }
                temp *= this.m_modelSecondDerivativesParameters
                    .getEntry(k);
                for (int i = 0; i < n; i++) {
                  this.m_gradientAtTrustRegionCenter.setEntry(i,
                      this.m_gradientAtTrustRegionCenter.getEntry(i)
                          + (temp * this.m_interpolationPoints.getEntry(k,
                              i)));
                }
              }
            }

            // Calculate the parameters of the least Frobenius norm
            // interpolant to
            // the current data, the gradient of this interpolant at XOPT
            // being put
            // into VLAG(NPT+I), I=1,2,...,N.

            if (ntrits > 0) {
              for (int k = 0; k < npt; k++) {
                this.m_lagrangeValuesAtNewPoint.setEntry(k,
                    this.m_fAtInterpolationPoints.getEntry(k)
                        - this.m_fAtInterpolationPoints.getEntry(
                            this.m_trustRegionCenterInterpolationPointIndex));
                work3.setEntry(k, __SafeBOBYQAOptimizer.ZERO);
              }
              for (int j = 0; j < nptm; j++) {
                double sum = __SafeBOBYQAOptimizer.ZERO;
                for (int k = 0; k < npt; k++) {
                  sum += this.m_zMatrix.getEntry(k, j)
                      * this.m_lagrangeValuesAtNewPoint.getEntry(k);
                }
                for (int k = 0; k < npt; k++) {
                  work3.setEntry(k, work3.getEntry(k)
                      + (sum * this.m_zMatrix.getEntry(k, j)));
                }
              }
              for (int k = 0; k < npt; k++) {
                double sum = __SafeBOBYQAOptimizer.ZERO;
                for (int j = 0; j < n; j++) {
                  sum += this.m_interpolationPoints.getEntry(k, j)
                      * this.m_trustRegionCenterOffset.getEntry(j);
                }
                work2.setEntry(k, work3.getEntry(k));
                work3.setEntry(k, sum * work3.getEntry(k));
              }
              double gqsq = __SafeBOBYQAOptimizer.ZERO;
              double gisq = __SafeBOBYQAOptimizer.ZERO;
              for (int i = 0; i < n; i++) {
                double sum = __SafeBOBYQAOptimizer.ZERO;
                for (int k = 0; k < npt; k++) {
                  sum += (this.m_bMatrix.getEntry(k, i)
                      * this.m_lagrangeValuesAtNewPoint.getEntry(k))
                      + (this.m_interpolationPoints.getEntry(k, i)
                          * work3.getEntry(k));
                }
                if (this.m_trustRegionCenterOffset
                    .getEntry(i) == this.m_lowerDifference.getEntry(i)) {
                  // Computing MIN
                  // Computing 2nd power
                  final double d1 = FastMath.min(
                      __SafeBOBYQAOptimizer.ZERO,
                      this.m_gradientAtTrustRegionCenter.getEntry(i));
                  gqsq += d1 * d1;
                  // Computing 2nd power
                  final double d2 = FastMath
                      .min(__SafeBOBYQAOptimizer.ZERO, sum);
                  gisq += d2 * d2;
                } else
                  if (this.m_trustRegionCenterOffset
                      .getEntry(i) == this.m_upperDifference.getEntry(i)) {
                    // Computing MAX
                    // Computing 2nd power
                    final double d1 = FastMath.max(
                        __SafeBOBYQAOptimizer.ZERO,
                        this.m_gradientAtTrustRegionCenter.getEntry(i));
                    gqsq += d1 * d1;
                    // Computing 2nd power
                    final double d2 = FastMath
                        .max(__SafeBOBYQAOptimizer.ZERO, sum);
                    gisq += d2 * d2;
                  } else {
                    // Computing 2nd power
                    final double d1 = this.m_gradientAtTrustRegionCenter
                        .getEntry(i);
                    gqsq += d1 * d1;
                    gisq += sum * sum;
                  }
                this.m_lagrangeValuesAtNewPoint.setEntry(npt + i, sum);
              }

              // Test whether to replace the new quadratic model by the
              // least Frobenius
              // norm interpolant, making the replacement if the test is
              // satisfied.

              ++itest;
              if (gqsq < (__SafeBOBYQAOptimizer.TEN * gisq)) {
                itest = 0;
              }
              if (itest >= 3) {
                for (int i = 0, max = FastMath.max(npt,
                    nh); i < max; i++) {
                  if (i < n) {
                    this.m_gradientAtTrustRegionCenter.setEntry(i,
                        this.m_lagrangeValuesAtNewPoint.getEntry(npt + i));
                  }
                  if (i < npt) {
                    this.m_modelSecondDerivativesParameters.setEntry(i,
                        work2.getEntry(i));
                  }
                  if (i < nh) {
                    this.m_modelSecondDerivativesValues.setEntry(i,
                        __SafeBOBYQAOptimizer.ZERO);
                  }
                  itest = 0;
                }
              }
            }

            // If a trust region step has provided a sufficient decrease in
            // F, then
            // branch for another trust region calculation. The case
            // NTRITS=0 occurs
            // when the new interpolation point was reached by an
            // alternative step.

            if (ntrits == 0) {
              state = 60;
              break;
            }
            if (f <= (fopt
                + (__SafeBOBYQAOptimizer.ONE_OVER_TEN * vquad))) {
              state = 60;
              break;
            }

            // Alternatively, find out if the interpolation points are
            // close enough
            // to the best point so far.

            // Computing MAX
            // Computing 2nd power
            final double d1 = __SafeBOBYQAOptimizer.TWO * delta;
            // Computing 2nd power
            final double d2 = __SafeBOBYQAOptimizer.TEN * rho;
            distsq = FastMath.max(d1 * d1, d2 * d2);
          }
          case 650: {

            knew = -1;
            for (int k = 0; k < npt; k++) {
              double sum = __SafeBOBYQAOptimizer.ZERO;
              for (int j = 0; j < n; j++) {
                // Computing 2nd power
                final double d1 = this.m_interpolationPoints.getEntry(k, j)
                    - this.m_trustRegionCenterOffset.getEntry(j);
                sum += d1 * d1;
              }
              if (sum > distsq) {
                knew = k;
                distsq = sum;
              }
            }

            // If KNEW is positive, then ALTMOV finds alternative new
            // positions for
            // the KNEW-th interpolation point within distance ADELT of
            // XOPT. It is
            // reached via label 90. Otherwise, there is a branch to label
            // 60 for
            // another trust region iteration, unless the calculations with
            // the
            // current RHO are complete.

            if (knew >= 0) {
              final double dist = FastMath.sqrt(distsq);
              if (ntrits == -1) {
                // Computing MIN
                delta = FastMath.min(
                    __SafeBOBYQAOptimizer.ONE_OVER_TEN * delta,
                    __SafeBOBYQAOptimizer.HALF * dist);
                if (delta <= (rho * 1.5)) {
                  delta = rho;
                }
              }
              ntrits = 0;
              // Computing MAX
              // Computing MIN
              final double d1 = FastMath
                  .min(__SafeBOBYQAOptimizer.ONE_OVER_TEN * dist, delta);
              adelt = FastMath.max(d1, rho);
              dsq = adelt * adelt;
              state = 90;
              break;
            }
            if (ntrits == -1) {
              state = 680;
              break;
            }
            if (ratio > __SafeBOBYQAOptimizer.ZERO) {
              state = 60;
              break;
            }
            if (FastMath.max(delta, dnorm) > rho) {
              state = 60;
              break;
            }

            // The calculations with the current value of RHO are complete.
            // Pick the
            // next values of RHO and DELTA.
          }
          case 680: {

            if (rho > this.m_stoppingTrustRegionRadius) {
              delta = __SafeBOBYQAOptimizer.HALF * rho;
              ratio = rho / this.m_stoppingTrustRegionRadius;
              if (ratio <= __SafeBOBYQAOptimizer.SIXTEEN) {
                rho = this.m_stoppingTrustRegionRadius;
              } else
                if (ratio <= __SafeBOBYQAOptimizer.TWO_HUNDRED_FIFTY) {
                  rho = FastMath.sqrt(ratio)
                      * this.m_stoppingTrustRegionRadius;
                } else {
                  rho *= __SafeBOBYQAOptimizer.ONE_OVER_TEN;
                }
              delta = FastMath.max(delta, rho);
              ntrits = 0;
              nfsav = this.getEvaluations();
              state = 60;
              break;
            }

            // Return from the calculation, after another Newton-Raphson
            // step, if
            // it is too short to have been tried before.

            if (ntrits == -1) {
              state = 360;
              break;
            }
          }
          case 720: {

            if (this.m_fAtInterpolationPoints.getEntry(
                this.m_trustRegionCenterInterpolationPointIndex) <= fsave) {
              for (int i = 0; i < n; i++) {
                // Computing MIN
                // Computing MAX
                final double d3 = lowerBound[i];
                final double d4 = this.m_originShift.getEntry(i)
                    + this.m_trustRegionCenterOffset.getEntry(i);
                final double d1 = FastMath.max(d3, d4);
                final double d2 = upperBound[i];
                this.m_currentBest.setEntry(i, FastMath.min(d1, d2));
                if (this.m_trustRegionCenterOffset
                    .getEntry(i) == this.m_lowerDifference.getEntry(i)) {
                  this.m_currentBest.setEntry(i, lowerBound[i]);
                }
                if (this.m_trustRegionCenterOffset
                    .getEntry(i) == this.m_upperDifference.getEntry(i)) {
                  this.m_currentBest.setEntry(i, upperBound[i]);
                }
              }
              f = this.m_fAtInterpolationPoints.getEntry(
                  this.m_trustRegionCenterInterpolationPointIndex);
            }
            return f;
          }
          default: {
            throw new MathIllegalStateException(
                LocalizedFormats.SIMPLE_MESSAGE, "bobyqb"); //$NON-NLS-1$
          }
        }
      }

      return Double.POSITIVE_INFINITY;
    } // bobyqb

    // ----------------------------------------------------------------------------------------

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param knew
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param adelt
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @return the array see
     *         {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     */
    private final double[] __altmov(final int knew, final double adelt) {

      final int n = this.m_currentBest.getDimension();
      final int npt = this.m_numberOfInterpolationPoints;

      final ArrayRealVector glag = new ArrayRealVector(n);
      final ArrayRealVector hcol = new ArrayRealVector(npt);

      final ArrayRealVector work1 = new ArrayRealVector(n);
      final ArrayRealVector work2 = new ArrayRealVector(n);

      for (int k = 0; k < npt; k++) {
        hcol.setEntry(k, __SafeBOBYQAOptimizer.ZERO);
      }
      for (int j = 0, max = npt - n - 1; j < max; j++) {
        final double tmp = this.m_zMatrix.getEntry(knew, j);
        for (int k = 0; k < npt; k++) {
          hcol.setEntry(k,
              hcol.getEntry(k) + (tmp * this.m_zMatrix.getEntry(k, j)));
        }
      }
      final double alpha = hcol.getEntry(knew);
      final double ha = __SafeBOBYQAOptimizer.HALF * alpha;

      for (int i = 0; i < n; i++) {
        glag.setEntry(i, this.m_bMatrix.getEntry(knew, i));
      }
      for (int k = 0; k < npt; k++) {
        double tmp = __SafeBOBYQAOptimizer.ZERO;
        for (int j = 0; j < n; j++) {
          tmp += this.m_interpolationPoints.getEntry(k, j)
              * this.m_trustRegionCenterOffset.getEntry(j);
        }
        tmp *= hcol.getEntry(k);
        for (int i = 0; i < n; i++) {
          glag.setEntry(i, glag.getEntry(i)
              + (tmp * this.m_interpolationPoints.getEntry(k, i)));
        }
      }

      double presav = __SafeBOBYQAOptimizer.ZERO;
      double step = Double.NaN;
      int ksav = 0;
      int ibdsav = 0;
      double stpsav = 0;
      for (int k = 0; k < npt; k++) {
        if (k == this.m_trustRegionCenterInterpolationPointIndex) {
          continue;
        }
        double dderiv = __SafeBOBYQAOptimizer.ZERO;
        double distsq = __SafeBOBYQAOptimizer.ZERO;
        for (int i = 0; i < n; i++) {
          final double tmp = this.m_interpolationPoints.getEntry(k, i)
              - this.m_trustRegionCenterOffset.getEntry(i);
          dderiv += glag.getEntry(i) * tmp;
          distsq += tmp * tmp;
        }
        double subd = adelt / FastMath.sqrt(distsq);
        double slbd = -subd;
        int ilbd = 0;
        int iubd = 0;
        final double sumin = FastMath.min(__SafeBOBYQAOptimizer.ONE, subd);

        for (int i = 0; i < n; i++) {
          final double tmp = this.m_interpolationPoints.getEntry(k, i)
              - this.m_trustRegionCenterOffset.getEntry(i);
          if (tmp > __SafeBOBYQAOptimizer.ZERO) {
            if ((slbd * tmp) < (this.m_lowerDifference.getEntry(i)
                - this.m_trustRegionCenterOffset.getEntry(i))) {
              slbd = (this.m_lowerDifference.getEntry(i)
                  - this.m_trustRegionCenterOffset.getEntry(i)) / tmp;
              ilbd = -i - 1;
            }
            if ((subd * tmp) > (this.m_upperDifference.getEntry(i)
                - this.m_trustRegionCenterOffset.getEntry(i))) {
              // Computing MAX
              subd = FastMath
                  .max(sumin,
                      (this.m_upperDifference.getEntry(i)
                          - this.m_trustRegionCenterOffset.getEntry(i))
                          / tmp);
              iubd = i + 1;
            }
          } else
            if (tmp < __SafeBOBYQAOptimizer.ZERO) {
              if ((slbd * tmp) > (this.m_upperDifference.getEntry(i)
                  - this.m_trustRegionCenterOffset.getEntry(i))) {
                slbd = (this.m_upperDifference.getEntry(i)
                    - this.m_trustRegionCenterOffset.getEntry(i)) / tmp;
                ilbd = i + 1;
              }
              if ((subd * tmp) < (this.m_lowerDifference.getEntry(i)
                  - this.m_trustRegionCenterOffset.getEntry(i))) {
                // Computing MAX
                subd = FastMath.max(sumin,
                    (this.m_lowerDifference.getEntry(i)
                        - this.m_trustRegionCenterOffset.getEntry(i))
                        / tmp);
                iubd = -i - 1;
              }
            }
        }

        step = slbd;
        int isbd = ilbd;
        double vlag = Double.NaN;
        if (k == knew) {
          final double diff = dderiv - __SafeBOBYQAOptimizer.ONE;
          vlag = slbd * (dderiv - (slbd * diff));
          final double d1 = subd * (dderiv - (subd * diff));
          if (FastMath.abs(d1) > FastMath.abs(vlag)) {
            step = subd;
            vlag = d1;
            isbd = iubd;
          }
          final double d2 = __SafeBOBYQAOptimizer.HALF * dderiv;
          final double d3 = d2 - (diff * slbd);
          final double d4 = d2 - (diff * subd);
          if ((d3 * d4) < __SafeBOBYQAOptimizer.ZERO) {
            final double d5 = (d2 * d2) / diff;
            if (FastMath.abs(d5) > FastMath.abs(vlag)) {
              step = d2 / diff;
              vlag = d5;
              isbd = 0;
            }
          }

        } else {
          vlag = slbd * (__SafeBOBYQAOptimizer.ONE - slbd);
          final double tmp = subd * (__SafeBOBYQAOptimizer.ONE - subd);
          if (FastMath.abs(tmp) > FastMath.abs(vlag)) {
            step = subd;
            vlag = tmp;
            isbd = iubd;
          }
          if ((subd > __SafeBOBYQAOptimizer.HALF) && (FastMath
              .abs(vlag) < __SafeBOBYQAOptimizer.ONE_OVER_FOUR)) {
            step = __SafeBOBYQAOptimizer.HALF;
            vlag = __SafeBOBYQAOptimizer.ONE_OVER_FOUR;
            isbd = 0;
          }
          vlag *= dderiv;
        }

        final double tmp = step * (__SafeBOBYQAOptimizer.ONE - step)
            * distsq;
        final double predsq = vlag * vlag
            * ((vlag * vlag) + (ha * tmp * tmp));
        if (predsq > presav) {
          presav = predsq;
          ksav = k;
          stpsav = step;
          ibdsav = isbd;
        }
      }

      for (int i = 0; i < n; i++) {
        final double tmp = this.m_trustRegionCenterOffset.getEntry(i)
            + (stpsav * (this.m_interpolationPoints.getEntry(ksav, i)
                - this.m_trustRegionCenterOffset.getEntry(i)));
        this.m_newPoint.setEntry(i,
            FastMath.max(this.m_lowerDifference.getEntry(i),
                FastMath.min(this.m_upperDifference.getEntry(i), tmp)));
      }
      if (ibdsav < 0) {
        this.m_newPoint.setEntry(-ibdsav - 1,
            this.m_lowerDifference.getEntry(-ibdsav - 1));
      }
      if (ibdsav > 0) {
        this.m_newPoint.setEntry(ibdsav - 1,
            this.m_upperDifference.getEntry(ibdsav - 1));
      }

      final double bigstp = adelt + adelt;
      int iflag = 0;
      double cauchy = Double.NaN;
      double csave = __SafeBOBYQAOptimizer.ZERO;
      while (true) {
        double wfixsq = __SafeBOBYQAOptimizer.ZERO;
        double ggfree = __SafeBOBYQAOptimizer.ZERO;
        for (int i = 0; i < n; i++) {
          final double glagValue = glag.getEntry(i);
          work1.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
          if ((FastMath.min(
              this.m_trustRegionCenterOffset.getEntry(i)
                  - this.m_lowerDifference.getEntry(i),
              glagValue) > __SafeBOBYQAOptimizer.ZERO)
              || (FastMath.max(
                  this.m_trustRegionCenterOffset.getEntry(i)
                      - this.m_upperDifference.getEntry(i),
                  glagValue) < __SafeBOBYQAOptimizer.ZERO)) {
            work1.setEntry(i, bigstp);
            ggfree += glagValue * glagValue;
          }
        }
        if (ggfree == __SafeBOBYQAOptimizer.ZERO) {
          return new double[] { alpha, __SafeBOBYQAOptimizer.ZERO };
        }

        final double tmp1 = (adelt * adelt) - wfixsq;
        if (tmp1 > __SafeBOBYQAOptimizer.ZERO) {
          step = FastMath.sqrt(tmp1 / ggfree);
          ggfree = __SafeBOBYQAOptimizer.ZERO;
          for (int i = 0; i < n; i++) {
            if (work1.getEntry(i) == bigstp) {
              final double tmp2 = this.m_trustRegionCenterOffset
                  .getEntry(i) - (step * glag.getEntry(i));
              if (tmp2 <= this.m_lowerDifference.getEntry(i)) {
                work1.setEntry(i, this.m_lowerDifference.getEntry(i)
                    - this.m_trustRegionCenterOffset.getEntry(i));
                final double d1 = work1.getEntry(i);
                wfixsq += d1 * d1;
              } else
                if (tmp2 >= this.m_upperDifference.getEntry(i)) {
                  work1.setEntry(i, this.m_upperDifference.getEntry(i)
                      - this.m_trustRegionCenterOffset.getEntry(i));
                  final double d1 = work1.getEntry(i);
                  wfixsq += d1 * d1;
                } else {
                  final double d1 = glag.getEntry(i);
                  ggfree += d1 * d1;
                }
            }
          }
        }

        double gw = __SafeBOBYQAOptimizer.ZERO;
        for (int i = 0; i < n; i++) {
          final double glagValue = glag.getEntry(i);
          if (work1.getEntry(i) == bigstp) {
            work1.setEntry(i, -step * glagValue);
            final double min = FastMath.min(
                this.m_upperDifference.getEntry(i),
                this.m_trustRegionCenterOffset.getEntry(i)
                    + work1.getEntry(i));
            this.m_alternativeNewPoint.setEntry(i,
                FastMath.max(this.m_lowerDifference.getEntry(i), min));
          } else
            if (work1.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
              this.m_alternativeNewPoint.setEntry(i,
                  this.m_trustRegionCenterOffset.getEntry(i));
            } else
              if (glagValue > __SafeBOBYQAOptimizer.ZERO) {
                this.m_alternativeNewPoint.setEntry(i,
                    this.m_lowerDifference.getEntry(i));
              } else {
                this.m_alternativeNewPoint.setEntry(i,
                    this.m_upperDifference.getEntry(i));
              }
          gw += glagValue * work1.getEntry(i);
        }

        double curv = __SafeBOBYQAOptimizer.ZERO;
        for (int k = 0; k < npt; k++) {
          double tmp = __SafeBOBYQAOptimizer.ZERO;
          for (int j = 0; j < n; j++) {
            tmp += this.m_interpolationPoints.getEntry(k, j)
                * work1.getEntry(j);
          }
          curv += hcol.getEntry(k) * tmp * tmp;
        }
        if (iflag == 1) {
          curv = -curv;
        }
        if ((curv > -gw) && (curv < (-gw * (__SafeBOBYQAOptimizer.ONE
            + FastMath.sqrt(__SafeBOBYQAOptimizer.TWO))))) {
          final double scale = -gw / curv;
          for (int i = 0; i < n; i++) {
            final double tmp = this.m_trustRegionCenterOffset.getEntry(i)
                + (scale * work1.getEntry(i));
            this.m_alternativeNewPoint.setEntry(i, FastMath.max(
                this.m_lowerDifference.getEntry(i),
                FastMath.min(this.m_upperDifference.getEntry(i), tmp)));
          }
          // Computing 2nd power
          final double d1 = __SafeBOBYQAOptimizer.HALF * gw * scale;
          cauchy = d1 * d1;
        } else {
          // Computing 2nd power
          final double d1 = gw + (__SafeBOBYQAOptimizer.HALF * curv);
          cauchy = d1 * d1;
        }

        if (iflag == 0) {
          for (int i = 0; i < n; i++) {
            glag.setEntry(i, -glag.getEntry(i));
            work2.setEntry(i, this.m_alternativeNewPoint.getEntry(i));
          }
          csave = cauchy;
          iflag = 1;
        } else {
          break;
        }
      }
      if (csave > cauchy) {
        for (int i = 0; i < n; i++) {
          this.m_alternativeNewPoint.setEntry(i, work2.getEntry(i));
        }
        cauchy = csave;
      }

      return new double[] { alpha, cauchy };
    }

    // ----------------------------------------------------------------------------------------

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param lowerBound
     *          Lower bounds.
     * @param upperBound
     *          Upper bounds.
     */
    private final void __prelim(final double[] lowerBound,
        final double[] upperBound) {

      final int n = this.m_currentBest.getDimension();
      final int npt = this.m_numberOfInterpolationPoints;
      final int ndim = this.m_bMatrix.getRowDimension();

      final double rhosq = this.m_initialTrustRegionRadius
          * this.m_initialTrustRegionRadius;
      final double recip = 1d / rhosq;
      final int np = n + 1;

      // Set XBASE to the initial vector of variables, and set the initial
      // elements of XPT, BMAT, HQ, PQ and ZMAT to zero.

      for (int j = 0; j < n; j++) {
        this.m_originShift.setEntry(j, this.m_currentBest.getEntry(j));
        for (int k = 0; k < npt; k++) {
          this.m_interpolationPoints.setEntry(k, j,
              __SafeBOBYQAOptimizer.ZERO);
        }
        for (int i = 0; i < ndim; i++) {
          this.m_bMatrix.setEntry(i, j, __SafeBOBYQAOptimizer.ZERO);
        }
      }
      for (int i = 0, max = (n * np) / 2; i < max; i++) {
        this.m_modelSecondDerivativesValues.setEntry(i,
            __SafeBOBYQAOptimizer.ZERO);
      }
      for (int k = 0; k < npt; k++) {
        this.m_modelSecondDerivativesParameters.setEntry(k,
            __SafeBOBYQAOptimizer.ZERO);
        for (int j = 0, max = npt - np; j < max; j++) {
          this.m_zMatrix.setEntry(k, j, __SafeBOBYQAOptimizer.ZERO);
        }
      }

      int ipt = 0;
      int jpt = 0;
      double fbeg = Double.NaN;
      do {
        final int nfm = this.getEvaluations();
        final int nfx = nfm - n;
        final int nfmm = nfm - 1;
        final int nfxm = nfx - 1;
        double stepa = 0;
        double stepb = 0;
        if (nfm <= (2 * n)) {
          if ((nfm >= 1) && (nfm <= n)) {
            stepa = this.m_initialTrustRegionRadius;
            if (this.m_upperDifference
                .getEntry(nfmm) == __SafeBOBYQAOptimizer.ZERO) {
              stepa = -stepa;
            }
            this.m_interpolationPoints.setEntry(nfm, nfmm, stepa);
          } else
            if (nfm > n) {
              stepa = this.m_interpolationPoints.getEntry(nfx, nfxm);
              stepb = -this.m_initialTrustRegionRadius;
              if (this.m_lowerDifference
                  .getEntry(nfxm) == __SafeBOBYQAOptimizer.ZERO) {
                stepb = FastMath.min(
                    __SafeBOBYQAOptimizer.TWO
                        * this.m_initialTrustRegionRadius,
                    this.m_upperDifference.getEntry(nfxm));
              }
              if (this.m_upperDifference
                  .getEntry(nfxm) == __SafeBOBYQAOptimizer.ZERO) {
                stepb = FastMath.max(
                    -__SafeBOBYQAOptimizer.TWO
                        * this.m_initialTrustRegionRadius,
                    this.m_lowerDifference.getEntry(nfxm));
              }
              this.m_interpolationPoints.setEntry(nfm, nfxm, stepb);
            }
        } else {
          final int tmp1 = (nfm - np) / n;
          jpt = nfm - (tmp1 * n) - n;
          ipt = jpt + tmp1;
          if (ipt > n) {
            final int tmp2 = jpt;
            jpt = ipt - n;
            ipt = tmp2;
          }
          final int iptMinus1 = ipt - 1;
          final int jptMinus1 = jpt - 1;
          this.m_interpolationPoints.setEntry(nfm, iptMinus1,
              this.m_interpolationPoints.getEntry(ipt, iptMinus1));
          this.m_interpolationPoints.setEntry(nfm, jptMinus1,
              this.m_interpolationPoints.getEntry(jpt, jptMinus1));
        }

        for (int j = 0; j < n; j++) {
          this.m_currentBest.setEntry(j,
              FastMath.min(
                  FastMath.max(lowerBound[j],
                      this.m_originShift.getEntry(j)
                          + this.m_interpolationPoints.getEntry(nfm, j)),
              upperBound[j]));
          if (this.m_interpolationPoints.getEntry(nfm,
              j) == this.m_lowerDifference.getEntry(j)) {
            this.m_currentBest.setEntry(j, lowerBound[j]);
          }
          if (this.m_interpolationPoints.getEntry(nfm,
              j) == this.m_upperDifference.getEntry(j)) {
            this.m_currentBest.setEntry(j, upperBound[j]);
          }
        }

        final double objectiveValue = this
            .computeObjectiveValue(this.m_currentBest.toArray());
        final double f = objectiveValue;
        final int numEval = this.getEvaluations(); // nfm + 1
        this.m_fAtInterpolationPoints.setEntry(nfm, f);

        if (numEval == 1) {
          fbeg = f;
          this.m_trustRegionCenterInterpolationPointIndex = 0;
        } else
          if (f < this.m_fAtInterpolationPoints
              .getEntry(this.m_trustRegionCenterInterpolationPointIndex)) {
            this.m_trustRegionCenterInterpolationPointIndex = nfm;
          }

        if (numEval <= ((2 * n) + 1)) {
          if ((numEval >= 2) && (numEval <= (n + 1))) {
            this.m_gradientAtTrustRegionCenter.setEntry(nfmm,
                (f - fbeg) / stepa);
            if (npt < (numEval + n)) {
              final double oneOverStepA = __SafeBOBYQAOptimizer.ONE
                  / stepa;
              this.m_bMatrix.setEntry(0, nfmm, -oneOverStepA);
              this.m_bMatrix.setEntry(nfm, nfmm, oneOverStepA);
              this.m_bMatrix.setEntry(npt + nfmm, nfmm,
                  -__SafeBOBYQAOptimizer.HALF * rhosq);
            }
          } else
            if (numEval >= (n + 2)) {
              final int ih = ((nfx * (nfx + 1)) / 2) - 1;
              final double tmp = (f - fbeg) / stepb;
              final double diff = stepb - stepa;
              this.m_modelSecondDerivativesValues.setEntry(ih,
                  (__SafeBOBYQAOptimizer.TWO * (tmp
                      - this.m_gradientAtTrustRegionCenter.getEntry(nfxm)))
                      / diff);
              this.m_gradientAtTrustRegionCenter.setEntry(nfxm,
                  ((this.m_gradientAtTrustRegionCenter.getEntry(nfxm)
                      * stepb) - (tmp * stepa)) / diff);
              if (((stepa * stepb) < __SafeBOBYQAOptimizer.ZERO)
                  && (f < this.m_fAtInterpolationPoints
                      .getEntry(nfm - n))) {
                this.m_fAtInterpolationPoints.setEntry(nfm,
                    this.m_fAtInterpolationPoints.getEntry(nfm - n));
                this.m_fAtInterpolationPoints.setEntry(nfm - n, f);
                if (this.m_trustRegionCenterInterpolationPointIndex == nfm) {
                  this.m_trustRegionCenterInterpolationPointIndex = nfm
                      - n;
                }
                this.m_interpolationPoints.setEntry(nfm - n, nfxm, stepb);
                this.m_interpolationPoints.setEntry(nfm, nfxm, stepa);
              }
              this.m_bMatrix.setEntry(0, nfxm,
                  -(stepa + stepb) / (stepa * stepb));
              this.m_bMatrix.setEntry(nfm, nfxm,
                  -__SafeBOBYQAOptimizer.HALF / this.m_interpolationPoints
                      .getEntry(nfm - n, nfxm));
              this.m_bMatrix.setEntry(nfm - n, nfxm,
                  -this.m_bMatrix.getEntry(0, nfxm)
                      - this.m_bMatrix.getEntry(nfm, nfxm));
              this.m_zMatrix.setEntry(0, nfxm,
                  FastMath.sqrt(__SafeBOBYQAOptimizer.TWO)
                      / (stepa * stepb));
              this.m_zMatrix.setEntry(nfm, nfxm,
                  FastMath.sqrt(__SafeBOBYQAOptimizer.HALF) / rhosq);
              this.m_zMatrix.setEntry(nfm - n, nfxm,
                  -this.m_zMatrix.getEntry(0, nfxm)
                      - this.m_zMatrix.getEntry(nfm, nfxm));
            }
        } else {
          this.m_zMatrix.setEntry(0, nfxm, recip);
          this.m_zMatrix.setEntry(nfm, nfxm, recip);
          this.m_zMatrix.setEntry(ipt, nfxm, -recip);
          this.m_zMatrix.setEntry(jpt, nfxm, -recip);

          final int ih = (((ipt * (ipt - 1)) / 2) + jpt) - 1;
          final double tmp = this.m_interpolationPoints.getEntry(nfm,
              ipt - 1) * this.m_interpolationPoints.getEntry(nfm, jpt - 1);
          this.m_modelSecondDerivativesValues.setEntry(ih,
              ((fbeg - this.m_fAtInterpolationPoints.getEntry(ipt)
                  - this.m_fAtInterpolationPoints.getEntry(jpt)) + f)
                  / tmp);
        }
      } while (this.getEvaluations() < npt);
    }

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param delta
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param gnew
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param xbdi
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param s
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param hs
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param hred
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @return the array, or {@code null} if too many iterations were
     *         consumed
     */
    @SuppressWarnings("fallthrough")
    private double[] __trsbox(final double delta,
        final ArrayRealVector gnew, final ArrayRealVector xbdi,
        final ArrayRealVector s, final ArrayRealVector hs,
        final ArrayRealVector hred) {

      final int n = this.m_currentBest.getDimension();
      final int npt = this.m_numberOfInterpolationPoints;

      double dsq = Double.NaN;
      double crvmin = Double.NaN;

      // Local variables
      double ds;
      int iu;
      double dhd, dhs, cth, shs, sth, ssq, beta = 0, sdec, blen;
      int iact = -1;
      int nact = 0;
      double angt = 0, qred;
      int isav;
      double temp = 0, xsav = 0, xsum = 0, angbd = 0, dredg = 0, sredg = 0;
      int iterc;
      double resid = 0, delsq = 0, ggsav = 0, tempa = 0, tempb = 0,
          redmax = 0, dredsq = 0, redsav = 0, gredsq = 0, rednew = 0;
      int itcsav = 0;
      double rdprev = 0, rdnext = 0, stplen = 0, stepsq = 0;
      int itermax = 0;

      iterc = 0;
      nact = 0;
      for (int i = 0; i < n; i++) {
        xbdi.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
        if (this.m_trustRegionCenterOffset
            .getEntry(i) <= this.m_lowerDifference.getEntry(i)) {
          if (this.m_gradientAtTrustRegionCenter
              .getEntry(i) >= __SafeBOBYQAOptimizer.ZERO) {
            xbdi.setEntry(i, __SafeBOBYQAOptimizer.MINUS_ONE);
          }
        } else
          if ((this.m_trustRegionCenterOffset
              .getEntry(i) >= this.m_upperDifference.getEntry(i))
              && (this.m_gradientAtTrustRegionCenter
                  .getEntry(i) <= __SafeBOBYQAOptimizer.ZERO)) {
            xbdi.setEntry(i, __SafeBOBYQAOptimizer.ONE);
          }
        if (xbdi.getEntry(i) != __SafeBOBYQAOptimizer.ZERO) {
          ++nact;
        }
        this.m_trialStepPoint.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
        gnew.setEntry(i, this.m_gradientAtTrustRegionCenter.getEntry(i));
      }
      delsq = delta * delta;
      qred = __SafeBOBYQAOptimizer.ZERO;
      crvmin = __SafeBOBYQAOptimizer.MINUS_ONE;

      int state = 20;
      for (int index = 10000; (--index) >= 0;) {
        switch (state) {
          case 20: {
            beta = __SafeBOBYQAOptimizer.ZERO;
          }

          case 30: {
            stepsq = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) != __SafeBOBYQAOptimizer.ZERO) {
                s.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
              } else
                if (beta == __SafeBOBYQAOptimizer.ZERO) {
                  s.setEntry(i, -gnew.getEntry(i));
                } else {
                  s.setEntry(i, (beta * s.getEntry(i)) - gnew.getEntry(i));
                }
              // Computing 2nd power
              final double d1 = s.getEntry(i);
              stepsq += d1 * d1;
            }
            if (stepsq == __SafeBOBYQAOptimizer.ZERO) {
              state = 190;
              break;
            }
            if (beta == __SafeBOBYQAOptimizer.ZERO) {
              gredsq = stepsq;
              itermax = (iterc + n) - nact;
            }
            if ((gredsq * delsq) <= (qred * 1e-4 * qred)) {
              state = 190;
              break;
            }

            state = 210;
            break;
          }
          case 50: {

            resid = delsq;
            ds = __SafeBOBYQAOptimizer.ZERO;
            shs = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                final double d1 = this.m_trialStepPoint.getEntry(i);
                resid -= d1 * d1;
                ds += s.getEntry(i) * this.m_trialStepPoint.getEntry(i);
                shs += s.getEntry(i) * hs.getEntry(i);
              }
            }
            if (resid <= __SafeBOBYQAOptimizer.ZERO) {
              state = 90;
              break;
            }
            temp = FastMath.sqrt((stepsq * resid) + (ds * ds));
            if (ds < __SafeBOBYQAOptimizer.ZERO) {
              blen = (temp - ds) / stepsq;
            } else {
              blen = resid / (temp + ds);
            }
            stplen = blen;
            if (shs > __SafeBOBYQAOptimizer.ZERO) {
              stplen = FastMath.min(blen, gredsq / shs);
            }

            iact = -1;
            for (int i = 0; i < n; i++) {
              if (s.getEntry(i) != __SafeBOBYQAOptimizer.ZERO) {
                xsum = this.m_trustRegionCenterOffset.getEntry(i)
                    + this.m_trialStepPoint.getEntry(i);
                if (s.getEntry(i) > __SafeBOBYQAOptimizer.ZERO) {
                  temp = (this.m_upperDifference.getEntry(i) - xsum)
                      / s.getEntry(i);
                } else {
                  temp = (this.m_lowerDifference.getEntry(i) - xsum)
                      / s.getEntry(i);
                }
                if (temp < stplen) {
                  stplen = temp;
                  iact = i;
                }
              }
            }

            sdec = __SafeBOBYQAOptimizer.ZERO;
            if (stplen > __SafeBOBYQAOptimizer.ZERO) {
              ++iterc;
              temp = shs / stepsq;
              if ((iact == -1) && (temp > __SafeBOBYQAOptimizer.ZERO)) {
                crvmin = FastMath.min(crvmin, temp);
                if (crvmin == __SafeBOBYQAOptimizer.MINUS_ONE) {
                  crvmin = temp;
                }
              }
              ggsav = gredsq;
              gredsq = __SafeBOBYQAOptimizer.ZERO;
              for (int i = 0; i < n; i++) {
                gnew.setEntry(i,
                    gnew.getEntry(i) + (stplen * hs.getEntry(i)));
                if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                  // Computing 2nd power
                  final double d1 = gnew.getEntry(i);
                  gredsq += d1 * d1;
                }
                this.m_trialStepPoint.setEntry(i,
                    this.m_trialStepPoint.getEntry(i)
                        + (stplen * s.getEntry(i)));
              }
              final double d1 = stplen
                  * (ggsav - (__SafeBOBYQAOptimizer.HALF * stplen * shs));
              sdec = FastMath.max(d1, __SafeBOBYQAOptimizer.ZERO);
              qred += sdec;
            }

            if (iact >= 0) {
              ++nact;
              xbdi.setEntry(iact, __SafeBOBYQAOptimizer.ONE);
              if (s.getEntry(iact) < __SafeBOBYQAOptimizer.ZERO) {
                xbdi.setEntry(iact, __SafeBOBYQAOptimizer.MINUS_ONE);
              }
              final double d1 = this.m_trialStepPoint.getEntry(iact);
              delsq -= d1 * d1;
              if (delsq <= __SafeBOBYQAOptimizer.ZERO) {
                state = 190;
                break;
              }
              state = 20;
              break;
            }

            if (stplen < blen) {
              if (iterc == itermax) {
                state = 190;
                break;
              }
              if (sdec <= (qred * .01)) {
                state = 190;
                break;
              }
              beta = gredsq / ggsav;
              state = 30;
              break;
            }
          }
          case 90: {
            crvmin = __SafeBOBYQAOptimizer.ZERO;
          }

          case 100: {
            if (nact >= (n - 1)) {
              state = 190;
              break;
            }
            dredsq = __SafeBOBYQAOptimizer.ZERO;
            dredg = __SafeBOBYQAOptimizer.ZERO;
            gredsq = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                double d1 = this.m_trialStepPoint.getEntry(i);
                dredsq += d1 * d1;
                dredg += this.m_trialStepPoint.getEntry(i)
                    * gnew.getEntry(i);
                d1 = gnew.getEntry(i);
                gredsq += d1 * d1;
                s.setEntry(i, this.m_trialStepPoint.getEntry(i));
              } else {
                s.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
              }
            }
            itcsav = iterc;
            state = 210;
            break;
          }
          case 120: {

            ++iterc;
            temp = (gredsq * dredsq) - (dredg * dredg);
            if (temp <= (qred * 1e-4 * qred)) {
              state = 190;
              break;
            }
            temp = FastMath.sqrt(temp);
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                s.setEntry(i, ((dredg * this.m_trialStepPoint.getEntry(i))
                    - (dredsq * gnew.getEntry(i))) / temp);
              } else {
                s.setEntry(i, __SafeBOBYQAOptimizer.ZERO);
              }
            }
            sredg = -temp;

            angbd = __SafeBOBYQAOptimizer.ONE;
            iact = -1;
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                tempa = (this.m_trustRegionCenterOffset.getEntry(i)
                    + this.m_trialStepPoint.getEntry(i))
                    - this.m_lowerDifference.getEntry(i);
                tempb = this.m_upperDifference.getEntry(i)
                    - this.m_trustRegionCenterOffset.getEntry(i)
                    - this.m_trialStepPoint.getEntry(i);
                if (tempa <= __SafeBOBYQAOptimizer.ZERO) {
                  ++nact;
                  xbdi.setEntry(i, __SafeBOBYQAOptimizer.MINUS_ONE);
                  state = 100;
                  break;
                } else
                  if (tempb <= __SafeBOBYQAOptimizer.ZERO) {
                    ++nact;
                    xbdi.setEntry(i, __SafeBOBYQAOptimizer.ONE);
                    state = 100;
                    break;
                  }
                double d1 = this.m_trialStepPoint.getEntry(i);
                final double d2 = s.getEntry(i);
                ssq = (d1 * d1) + (d2 * d2);
                d1 = this.m_trustRegionCenterOffset.getEntry(i)
                    - this.m_lowerDifference.getEntry(i);
                temp = ssq - (d1 * d1);
                if (temp > __SafeBOBYQAOptimizer.ZERO) {
                  temp = FastMath.sqrt(temp) - s.getEntry(i);
                  if ((angbd * temp) > tempa) {
                    angbd = tempa / temp;
                    iact = i;
                    xsav = __SafeBOBYQAOptimizer.MINUS_ONE;
                  }
                }
                d1 = this.m_upperDifference.getEntry(i)
                    - this.m_trustRegionCenterOffset.getEntry(i);
                temp = ssq - (d1 * d1);
                if (temp > __SafeBOBYQAOptimizer.ZERO) {
                  temp = FastMath.sqrt(temp) + s.getEntry(i);
                  if ((angbd * temp) > tempb) {
                    angbd = tempb / temp;
                    iact = i;
                    xsav = __SafeBOBYQAOptimizer.ONE;
                  }
                }
              }
            }

            state = 210;
            break;
          }
          case 150: {

            shs = __SafeBOBYQAOptimizer.ZERO;
            dhs = __SafeBOBYQAOptimizer.ZERO;
            dhd = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                shs += s.getEntry(i) * hs.getEntry(i);
                dhs += this.m_trialStepPoint.getEntry(i) * hs.getEntry(i);
                dhd += this.m_trialStepPoint.getEntry(i)
                    * hred.getEntry(i);
              }
            }

            redmax = __SafeBOBYQAOptimizer.ZERO;
            isav = -1;
            redsav = __SafeBOBYQAOptimizer.ZERO;
            iu = (int) ((angbd * 17.) + 3.1);
            for (int i = 0; i < iu; i++) {
              angt = (angbd * i) / iu;
              sth = (angt + angt)
                  / (__SafeBOBYQAOptimizer.ONE + (angt * angt));
              temp = shs + (angt * ((angt * dhd) - dhs - dhs));
              rednew = sth * ((angt * dredg) - sredg
                  - (__SafeBOBYQAOptimizer.HALF * sth * temp));
              if (rednew > redmax) {
                redmax = rednew;
                isav = i;
                rdprev = redsav;
              } else
                if (i == (isav + 1)) {
                  rdnext = rednew;
                }
              redsav = rednew;
            }

            if (isav < 0) {
              state = 190;
              break;
            }
            if (isav < iu) {
              temp = (rdnext - rdprev)
                  / ((redmax + redmax) - rdprev - rdnext);
              angt = (angbd * (isav + (__SafeBOBYQAOptimizer.HALF * temp)))
                  / iu;
            }
            cth = (__SafeBOBYQAOptimizer.ONE - (angt * angt))
                / (__SafeBOBYQAOptimizer.ONE + (angt * angt));
            sth = (angt + angt)
                / (__SafeBOBYQAOptimizer.ONE + (angt * angt));
            temp = shs + (angt * ((angt * dhd) - dhs - dhs));
            sdec = sth * ((angt * dredg) - sredg
                - (__SafeBOBYQAOptimizer.HALF * sth * temp));
            if (sdec <= __SafeBOBYQAOptimizer.ZERO) {
              state = 190;
              break;
            }

            dredg = __SafeBOBYQAOptimizer.ZERO;
            gredsq = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              gnew.setEntry(i, gnew.getEntry(i)
                  + ((cth - __SafeBOBYQAOptimizer.ONE) * hred.getEntry(i))
                  + (sth * hs.getEntry(i)));
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ZERO) {
                this.m_trialStepPoint.setEntry(i,
                    (cth * this.m_trialStepPoint.getEntry(i))
                        + (sth * s.getEntry(i)));
                dredg += this.m_trialStepPoint.getEntry(i)
                    * gnew.getEntry(i);
                // Computing 2nd power
                final double d1 = gnew.getEntry(i);
                gredsq += d1 * d1;
              }
              hred.setEntry(i,
                  (cth * hred.getEntry(i)) + (sth * hs.getEntry(i)));
            }
            qred += sdec;
            if ((iact >= 0) && (isav == iu)) {
              ++nact;
              xbdi.setEntry(iact, xsav);
              state = 100;
              break;
            }

            if (sdec > (qred * .01)) {
              state = 120;
              break;
            }
          }
          case 190: {

            dsq = __SafeBOBYQAOptimizer.ZERO;
            for (int i = 0; i < n; i++) {
              final double min = FastMath.min(
                  this.m_trustRegionCenterOffset.getEntry(i)
                      + this.m_trialStepPoint.getEntry(i),
                  this.m_upperDifference.getEntry(i));
              this.m_newPoint.setEntry(i,
                  FastMath.max(min, this.m_lowerDifference.getEntry(i)));
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.MINUS_ONE) {
                this.m_newPoint.setEntry(i,
                    this.m_lowerDifference.getEntry(i));
              }
              if (xbdi.getEntry(i) == __SafeBOBYQAOptimizer.ONE) {
                this.m_newPoint.setEntry(i,
                    this.m_upperDifference.getEntry(i));
              }
              this.m_trialStepPoint.setEntry(i, this.m_newPoint.getEntry(i)
                  - this.m_trustRegionCenterOffset.getEntry(i));
              final double d1 = this.m_trialStepPoint.getEntry(i);
              dsq += d1 * d1;
            }
            return new double[] { dsq, crvmin };
          }

          case 210: {
            int ih = 0;
            for (int j = 0; j < n; j++) {
              hs.setEntry(j, __SafeBOBYQAOptimizer.ZERO);
              for (int i = 0; i <= j; i++) {
                if (i < j) {
                  hs.setEntry(j, hs.getEntry(j)
                      + (this.m_modelSecondDerivativesValues.getEntry(ih)
                          * s.getEntry(i)));
                }
                hs.setEntry(i,
                    hs.getEntry(i)
                        + (this.m_modelSecondDerivativesValues.getEntry(ih)
                            * s.getEntry(j)));
                ih++;
              }
            }
            final RealVector tmp = this.m_interpolationPoints.operate(s)
                .ebeMultiply(this.m_modelSecondDerivativesParameters);
            for (int k = 0; k < npt; k++) {
              if (this.m_modelSecondDerivativesParameters
                  .getEntry(k) != __SafeBOBYQAOptimizer.ZERO) {
                for (int i = 0; i < n; i++) {
                  hs.setEntry(i, hs.getEntry(i) + (tmp.getEntry(k)
                      * this.m_interpolationPoints.getEntry(k, i)));
                }
              }
            }
            if (crvmin != __SafeBOBYQAOptimizer.ZERO) {
              state = 50;
              break;
            }
            if (iterc > itcsav) {
              state = 150;
              break;
            }
            for (int i = 0; i < n; i++) {
              hred.setEntry(i, hs.getEntry(i));
            }
            state = 120;
            break;
          }
          default: {
            throw new MathIllegalStateException(
                LocalizedFormats.SIMPLE_MESSAGE, "trsbox"); //$NON-NLS-1$
          }
        }
      }

      return null;// too many iterations
    }

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param beta
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param denom
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     * @param knew
     *          see
     *          {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     */
    private final void __update(final double beta, final double denom,
        final int knew) {

      final int n = this.m_currentBest.getDimension();
      final int npt = this.m_numberOfInterpolationPoints;
      final int nptm = npt - n - 1;

      final ArrayRealVector work = new ArrayRealVector(npt + n);

      double ztest = __SafeBOBYQAOptimizer.ZERO;
      for (int k = 0; k < npt; k++) {
        for (int j = 0; j < nptm; j++) {
          // Computing MAX
          ztest = FastMath.max(ztest,
              FastMath.abs(this.m_zMatrix.getEntry(k, j)));
        }
      }
      ztest *= 1e-20;

      for (int j = 1; j < nptm; j++) {
        final double d1 = this.m_zMatrix.getEntry(knew, j);
        if (FastMath.abs(d1) > ztest) {
          // Computing 2nd power
          final double d2 = this.m_zMatrix.getEntry(knew, 0);
          // Computing 2nd power
          final double d3 = this.m_zMatrix.getEntry(knew, j);
          final double d4 = FastMath.sqrt((d2 * d2) + (d3 * d3));
          final double d5 = this.m_zMatrix.getEntry(knew, 0) / d4;
          final double d6 = this.m_zMatrix.getEntry(knew, j) / d4;
          for (int i = 0; i < npt; i++) {
            final double d7 = (d5 * this.m_zMatrix.getEntry(i, 0))
                + (d6 * this.m_zMatrix.getEntry(i, j));
            this.m_zMatrix.setEntry(i, j,
                (d5 * this.m_zMatrix.getEntry(i, j))
                    - (d6 * this.m_zMatrix.getEntry(i, 0)));
            this.m_zMatrix.setEntry(i, 0, d7);
          }
        }
        this.m_zMatrix.setEntry(knew, j, __SafeBOBYQAOptimizer.ZERO);
      }

      for (int i = 0; i < npt; i++) {
        work.setEntry(i, this.m_zMatrix.getEntry(knew, 0)
            * this.m_zMatrix.getEntry(i, 0));
      }
      final double alpha = work.getEntry(knew);
      final double tau = this.m_lagrangeValuesAtNewPoint.getEntry(knew);
      this.m_lagrangeValuesAtNewPoint.setEntry(knew,
          this.m_lagrangeValuesAtNewPoint.getEntry(knew)
              - __SafeBOBYQAOptimizer.ONE);

      // Complete the updating of ZMAT.

      final double sqrtDenom = FastMath.sqrt(denom);
      final double d1 = tau / sqrtDenom;
      final double d2 = this.m_zMatrix.getEntry(knew, 0) / sqrtDenom;
      for (int i = 0; i < npt; i++) {
        this.m_zMatrix.setEntry(i, 0, (d1 * this.m_zMatrix.getEntry(i, 0))
            - (d2 * this.m_lagrangeValuesAtNewPoint.getEntry(i)));
      }

      // Finally, update the matrix BMAT.

      for (int j = 0; j < n; j++) {
        final int jp = npt + j;
        work.setEntry(jp, this.m_bMatrix.getEntry(knew, j));
        final double d3 = ((alpha
            * this.m_lagrangeValuesAtNewPoint.getEntry(jp))
            - (tau * work.getEntry(jp))) / denom;
        final double d4 = ((-beta * work.getEntry(jp))
            - (tau * this.m_lagrangeValuesAtNewPoint.getEntry(jp)))
            / denom;
        for (int i = 0; i <= jp; i++) {
          this.m_bMatrix.setEntry(i, j,
              this.m_bMatrix.getEntry(i, j)
                  + (d3 * this.m_lagrangeValuesAtNewPoint.getEntry(i))
                  + (d4 * work.getEntry(i)));
          if (i >= npt) {
            this.m_bMatrix.setEntry(jp, (i - npt),
                this.m_bMatrix.getEntry(i, j));
          }
        }
      }
    }

    /**
     * see
     * {@link org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer}
     *
     * @param lowerBound
     *          Lower bounds (constraints) of the objective variables.
     * @param upperBound
     *          Upperer bounds (constraints) of the objective variables.
     */
    private final void __setup(final double[] lowerBound,
        final double[] upperBound) {

      final double[] init = this.getStartPoint();
      final int dimension = init.length;

      // Check problem dimension.
      if (dimension < __SafeBOBYQAOptimizer.MINIMUM_PROBLEM_DIMENSION) {
        throw new NumberIsTooSmallException(Integer.valueOf(dimension),
            Integer.valueOf(
                __SafeBOBYQAOptimizer.MINIMUM_PROBLEM_DIMENSION),
            true);
      }
      // Check number of interpolation points.
      final int[] nPointsInterval = { dimension + 2,
          ((dimension + 2) * (dimension + 1)) / 2 };
      if ((this.m_numberOfInterpolationPoints < nPointsInterval[0])
          || (this.m_numberOfInterpolationPoints > nPointsInterval[1])) {
        throw new OutOfRangeException(
            LocalizedFormats.NUMBER_OF_INTERPOLATION_POINTS,
            Integer.valueOf(this.m_numberOfInterpolationPoints),
            Integer.valueOf(nPointsInterval[0]),
            Integer.valueOf(nPointsInterval[1]));
      }

      // Initialize bound differences.
      this.m_boundDifference = new double[dimension];

      final double requiredMinDiff = 2 * this.m_initialTrustRegionRadius;
      double minDiff = Double.POSITIVE_INFINITY;
      for (int i = 0; i < dimension; i++) {
        this.m_boundDifference[i] = upperBound[i] - lowerBound[i];
        minDiff = FastMath.min(minDiff, this.m_boundDifference[i]);
      }
      if (minDiff < requiredMinDiff) {
        this.m_initialTrustRegionRadius = minDiff / 3.0;
      }

      // Initialize the data structures used by the "bobyqa" method.
      this.m_bMatrix = new Array2DRowRealMatrix(
          dimension + this.m_numberOfInterpolationPoints, dimension);
      this.m_zMatrix = new Array2DRowRealMatrix(
          this.m_numberOfInterpolationPoints,
          this.m_numberOfInterpolationPoints - dimension - 1);
      this.m_interpolationPoints = new Array2DRowRealMatrix(
          this.m_numberOfInterpolationPoints, dimension);
      this.m_originShift = new ArrayRealVector(dimension);
      this.m_fAtInterpolationPoints = new ArrayRealVector(
          this.m_numberOfInterpolationPoints);
      this.m_trustRegionCenterOffset = new ArrayRealVector(dimension);
      this.m_gradientAtTrustRegionCenter = new ArrayRealVector(dimension);
      this.m_lowerDifference = new ArrayRealVector(dimension);
      this.m_upperDifference = new ArrayRealVector(dimension);
      this.m_modelSecondDerivativesParameters = new ArrayRealVector(
          this.m_numberOfInterpolationPoints);
      this.m_newPoint = new ArrayRealVector(dimension);
      this.m_alternativeNewPoint = new ArrayRealVector(dimension);
      this.m_trialStepPoint = new ArrayRealVector(dimension);
      this.m_lagrangeValuesAtNewPoint = new ArrayRealVector(
          dimension + this.m_numberOfInterpolationPoints);
      this.m_modelSecondDerivativesValues = new ArrayRealVector(
          (dimension * (dimension + 1)) / 2);
    }
  }
}
