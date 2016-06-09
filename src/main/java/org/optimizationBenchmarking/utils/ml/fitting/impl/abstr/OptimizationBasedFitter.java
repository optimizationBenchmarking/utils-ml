package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import org.optimizationBenchmarking.utils.reflection.ReflectionUtils;

/** A curve fitter that uses {@link OptimizationBasedFittingJob}s. */
public abstract class OptimizationBasedFitter extends FunctionFitter {

  /** the error */
  private final Throwable m_error;

  /** create */
  protected OptimizationBasedFitter() {
    super();

    Throwable cannot;

    cannot = null;
    try {
      ReflectionUtils.ensureClassesAreLoaded(//
          "org.apache.commons.math3.analysis.MultivariateFunction", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.linear.Array2DRowRealMatrix", //$NON-NLS-1$
          "org.apache.commons.math3.linear.ArrayRealVector", //$NON-NLS-1$
          "org.apache.commons.math3.linear.RealVector", //$NON-NLS-1$
          "org.apache.commons.math3.optim.ConvergenceChecker", //$NON-NLS-1$
          "org.apache.commons.math3.optim.InitialGuess", //$NON-NLS-1$
          "org.apache.commons.math3.optim.MaxEval", //$NON-NLS-1$
          "org.apache.commons.math3.optim.MaxIter", //$NON-NLS-1$
          "org.apache.commons.math3.optim.PointValuePair", //$NON-NLS-1$
          "org.apache.commons.math3.optim.SimpleBounds", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.GoalType", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.util.Incrementor" //$NON-NLS-1$
      );

    } catch (final Throwable error) {
      cannot = error;
    }

    this.m_error = cannot;
  }

  /** {@inheritDoc} */
  @Override
  public boolean canUse() {
    return (this.m_error == null);
  }

  /** {@inheritDoc} */
  @Override
  public void checkCanUse() {
    if (this.m_error != null) {
      throw new UnsupportedOperationException(//
          this.toString()
              + " cannot be used since required classes are missing in the classpath.", //$NON-NLS-1$
          this.m_error);
    }
    super.checkCanUse();
  }
}
