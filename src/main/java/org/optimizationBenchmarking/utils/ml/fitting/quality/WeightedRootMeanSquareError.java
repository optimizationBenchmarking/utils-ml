package org.optimizationBenchmarking.utils.ml.fitting.quality;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.StableSum;
import org.optimizationBenchmarking.utils.ml.fitting.spec.FittingEvaluation;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

/**
 * A quality measure which attempts to give each point the same influence
 * on the fitting outcome. This is achieved by weighting each residuals by
 * the inverse of the corresponding absolute value of the expected output
 * values. Basically, we consider the absolute values of {@code y}
 * -coordinates of points as their inverse weight. This way, if the
 * estimate for a point with {@code y}-value {@code 10} is {@code 11}, this
 * has the same impact as if the estimate of a point with {@code y}-value
 * {@code -1e-10} is {@code -1.1e-10}. Of course, we need to cater for
 * points with {@code y}-value {@code 0}, for which we choose a reasonable
 * minimum weight limit.
 */
public final class WeightedRootMeanSquareError
    extends FittingQualityMeasure {

  /** a stable sum, only to be used during solution evaluation */
  private final StableSum m_sum;

  /**
   * The minimum inverse weight: Basically, we consider the absolute values
   * of {@code y}-coordinates of points as their inverse weight. This way,
   * if the estimate for a point with {@code y}-value {@code 10} is
   * {@code 11}, this has the same impact as if the estimate of a point
   * with {@code y}-value {@code -1e-10} is {@code -1.1e-10}. Of course, we
   * need to cater for points with {@code y}-value {@code 0}, which will
   * then have this inverse weight here.
   */
  private final double m_minInverseWeight;

  /**
   * create the root-mean-square error fitting quality measure
   *
   * @param data
   *          the data matrix
   */
  public WeightedRootMeanSquareError(final IMatrix data) {
    super(data);

    int index;
    double currentY, minY, minY2;

    // find the two smallest non-zero absolute y values
    minY = minY2 = Double.POSITIVE_INFINITY;
    for (index = data.m(); (--index) >= 0;) {
      currentY = Math.abs(data.getDouble(index, 1));
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        if (currentY < minY2) {
          if (currentY < minY) {
            minY = currentY;
          } else {
            if (currentY > minY) {
              minY2 = currentY;
            }
          }
        }
      }
    }

    this.m_sum = new StableSum();

    // define the weight of points with "0" y-coordinate
    if (WeightedRootMeanSquareError.__checkInverseWeight(minY)) {

      if (WeightedRootMeanSquareError.__checkInverseWeight(minY2)) {
        // Ideally, the inverse weight of the point with 0 y-value
        // behaves to the weight of the point with next-larger (i.e.,
        // smallest non-zero) absolute y value like this point's weight
        // to the one with second-smallest non-zero absolute y value.
        currentY = (minY * (minY / minY2));
        if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
          this.m_minInverseWeight = currentY;
          return;
        }
      }

      // If that is not possible, we say it should just be ten percent
      // smaller.
      currentY = (minY * 0.9d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        this.m_minInverseWeight = currentY;
        return;
      }

      // If that is not possible, we say it should just be one percent
      // smaller.
      currentY = (minY * 0.99d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        this.m_minInverseWeight = currentY;
        return;
      }

      // If that is not possible, we say it should just be 0.1 percent
      // smaller.
      currentY = (minY * 0.999d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        this.m_minInverseWeight = currentY;
        return;
      }

      // If that is not possible, we say it should just be the tiniest
      // bit smaller.
      currentY = Math.nextAfter(minY, Double.NEGATIVE_INFINITY);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        this.m_minInverseWeight = currentY;
        return;
      }

      // If that is not possible (smallest y-coordinate is
      // |Double.MIN_NORMAL|?), we say it should just be the same.
      this.m_minInverseWeight = minY;
      return;
    }

    // If that is not possible (all weights are <=Double.MIN_NORMAL????),
    // we say it should just be 1.
    this.m_minInverseWeight = 1d;
  }

  /**
   * check the given inverse weight
   *
   * @param w
   *          the weight
   * @return {@code true} if it can be used, {@code false} otherwise
   */
  private static final boolean __checkInverseWeight(final double w) {
    return ((w > Double.MIN_NORMAL) && (w < Double.POSITIVE_INFINITY));
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final ParametricUnaryFunction model,
      final double[] params) {
    final StableSum sum;
    final double minInverseWeight;
    final int length;
    double expectedY, computedY, residual;
    int index;

    sum = this.m_sum;
    sum.reset();

    minInverseWeight = this.m_minInverseWeight;
    length = this.m_data.m();
    for (index = length; (--index) >= 0;) {
      expectedY = this.m_data.getDouble(index, 1);
      computedY = model.value(this.m_data.getDouble(index, 0), params);
      residual = (expectedY - computedY);
      if (residual != 0d) {
        residual /= ((expectedY < minInverseWeight) ? minInverseWeight
            : expectedY);
      }
      sum.append(residual * residual);
    }

    residual = Math.sqrt(sum.doubleValue() / length);
    return (MathUtils.isFinite(residual) ? residual
        : Double.POSITIVE_INFINITY);
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluateAt(final ParametricUnaryFunction model,
      final double[] params, final int... points) {
    final StableSum sum;
    final double minInverseWeight;
    double expectedY, computedY, residual;

    sum = this.m_sum;
    sum.reset();

    minInverseWeight = this.m_minInverseWeight;
    for (final int index : points) {
      expectedY = this.m_data.getDouble(index, 1);
      computedY = model.value(this.m_data.getDouble(index, 0), params);
      residual = (expectedY - computedY);
      if (residual != 0d) {
        residual /= ((expectedY < minInverseWeight) ? minInverseWeight
            : expectedY);
      }
      sum.append(residual * residual);
    }

    residual = Math.sqrt(sum.doubleValue() / points.length);
    return (MathUtils.isFinite(residual) ? residual
        : Double.POSITIVE_INFINITY);
  }

  /** {@inheritDoc} */
  @Override
  public final void evaluate(final ParametricUnaryFunction model,
      final double[] parameters, final FittingEvaluation dest) {
    double[][] jacobian;
    double[] residuals;
    final int numSamples, numParams;
    final IMatrix data;
    final double minInverseWeight;
    final StableSum sum;
    double[] jacobianRow;
    double x, expectedY, computedY, residual, inverseWeight,
        squareErrorSum;
    int index, j;

    data = this.m_data;
    minInverseWeight = this.m_minInverseWeight;

    numSamples = data.m();

    residuals = dest.residuals;
    if ((residuals == null) || (residuals.length != numSamples)) {
      dest.residuals = residuals = new double[numSamples];
    }

    numParams = parameters.length;// =model.getParameterCount();
    jacobian = dest.jacobian;
    if ((jacobian == null) || (jacobian.length != numSamples)
        || (jacobian[0].length != numParams)) {
      dest.jacobian = jacobian = new double[numSamples][numParams];
    }
    sum = this.m_sum;

    sum.reset();
    for (index = numSamples; (--index) >= 0;) {
      x = data.getDouble(index, 0);
      expectedY = this.m_data.getDouble(index, 1);
      computedY = model.value(x, parameters);
      if (expectedY < minInverseWeight) {
        inverseWeight = minInverseWeight;
      } else {
        inverseWeight = expectedY;
      }
      residual = ((expectedY - computedY) / inverseWeight);

      residuals[index] = residual;
      sum.append(residual * residual);

      jacobianRow = jacobian[index];
      model.gradient(x, parameters, jacobianRow);
      for (j = numParams; (--j) >= 0;) {
        jacobianRow[j] /= inverseWeight;
      }
    }

    squareErrorSum = sum.doubleValue();
    if (MathUtils.isFinite(squareErrorSum)) {
      dest.rmsError = dest.quality = //
      Math.sqrt(squareErrorSum / numSamples);
      dest.rsError = Math.sqrt(squareErrorSum);
    } else {
      dest.rmsError = dest.rsError = dest.quality = Double.POSITIVE_INFINITY;
    }
  }

  /** {@inheritDoc} */
  @Override
  public final void evaluateAt(final ParametricUnaryFunction model,
      final double[] parameters, final FittingEvaluation dest,
      final int... points) {
    double[][] jacobian;
    double[] residuals;
    final int numSamples, numParams;
    final IMatrix data;
    final double minInverseWeight;
    final StableSum sum;
    double[] jacobianRow;
    double x, expectedY, computedY, residual, inverseWeight,
        squareErrorSum;
    int j, index;

    data = this.m_data;
    minInverseWeight = this.m_minInverseWeight;

    numSamples = points.length;

    residuals = dest.residuals;
    if ((residuals == null) || (residuals.length != numSamples)) {
      dest.residuals = residuals = new double[numSamples];
    }

    numParams = parameters.length;// =model.getParameterCount();
    jacobian = dest.jacobian;
    if ((jacobian == null) || (jacobian.length != numSamples)
        || (jacobian[0].length != numParams)) {
      dest.jacobian = jacobian = new double[numSamples][numParams];
    }
    sum = this.m_sum;

    sum.reset();
    index=(-1);
    for (final int point : points) {
      x = data.getDouble(point, 0);
      expectedY = this.m_data.getDouble(point, 1);
      computedY = model.value(x, parameters);
      if (expectedY < minInverseWeight) {
        inverseWeight = minInverseWeight;
      } else {
        inverseWeight = expectedY;
      }
      residual = ((expectedY - computedY) / inverseWeight);

      ++index;
      residuals[index] = residual;
      sum.append(residual * residual);

      jacobianRow = jacobian[index];
      model.gradient(x, parameters, jacobianRow);
      for (j = numParams; (--j) >= 0;) {
        jacobianRow[j] /= inverseWeight;
      }
    }

    squareErrorSum = sum.doubleValue();
    if (MathUtils.isFinite(squareErrorSum)) {
      dest.rmsError = dest.quality = //
      Math.sqrt(squareErrorSum / numSamples);
      dest.rsError = Math.sqrt(squareErrorSum);
    } else {
      dest.rmsError = dest.rsError = dest.quality = Double.POSITIVE_INFINITY;
    }
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "Weighted Root-Mean-Squared Error"; //$NON-NLS-1$
  }
}
