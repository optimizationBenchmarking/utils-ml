package org.optimizationBenchmarking.utils.ml.fitting.quality;

import java.util.Random;

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

  /** the values backing this quality measure */
  private final double[] m_values;

  /**
   * create the root-mean-square error fitting quality measure
   *
   * @param data
   *          the data matrix
   */
  public WeightedRootMeanSquareError(final IMatrix data) {
    this(WeightedRootMeanSquareError.__computeMatrix(data));
  }

  /**
   * create the root-mean-square error fitting quality measure
   *
   * @param data
   *          the data matrix
   */
  private WeightedRootMeanSquareError(final double[] data) {
    super();

    if ((data == null) || (data.length <= 0) || ((data.length % 3) != 0)) {
      throw new IllegalArgumentException(//
          "Invalid values array: must not be null or empty and must have a length which is a multiple of 3."); //$NON-NLS-1$
    }
    this.m_values = data;
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final ParametricUnaryFunction model,
      final double[] parameters) {
    final StableSum sum;
    final double[] data;
    double residual;
    int index;

    sum = new StableSum();
    sum.reset();

    data = this.m_values;
    for (index = data.length; index > 0;) {
      residual = ((model.value(data[--index], parameters) - data[--index])
          / data[--index]);
      sum.append(residual * residual);
    }

    residual = Math.sqrt(sum.doubleValue() / (data.length / 3));
    return (MathUtils.isFinite(residual) ? residual
        : Double.POSITIVE_INFINITY);
  }

  /** {@inheritDoc} */
  @Override
  public final void evaluate(final ParametricUnaryFunction model,
      final double[] parameters, final boolean computeResiduals,
      final boolean computeJacobinian, final FittingEvaluation dest) {
    double[][] jacobian;
    double[] residuals;
    final int numSamples, numParams;
    final double[] data;
    final StableSum sum;
    double[] jacobianRow;
    double x, expectedY, residual, inverseWeight, squareErrorSum;
    int dataIndex, pointIndex, j;

    data = this.m_values;
    numSamples = (data.length / 3);

    residuals = dest.residuals;
    if (computeResiduals) {
      if ((residuals == null) || (residuals.length != numSamples)) {
        dest.residuals = residuals = new double[numSamples];
      }
    }

    numParams = parameters.length;// =model.getParameterCount();
    jacobian = dest.jacobian;
    if (computeJacobinian) {
      if ((jacobian == null) || (jacobian.length != numSamples)
          || (jacobian[0].length != numParams)) {
        dest.jacobian = jacobian = new double[numSamples][numParams];
      }
    }

    sum = new StableSum();
    sum.reset();

    for (dataIndex = data.length, pointIndex = numSamples; (--pointIndex) >= 0;) {
      x = data[--dataIndex];
      expectedY = data[--dataIndex];
      inverseWeight = data[--dataIndex];

      residual = ((expectedY - model.value(x, parameters))
          / inverseWeight);

      if (computeResiduals) {
        residuals[pointIndex] = residual;
      }
      sum.append(residual * residual);

      if (computeJacobinian) {
        jacobianRow = jacobian[pointIndex];
        model.gradient(x, parameters, jacobianRow);
        for (j = numParams; (--j) >= 0;) {
          jacobianRow[j] /= inverseWeight;
        }
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
  public final WeightedRootMeanSquareError subselect(final int npoints,
      final Random random) {
    final double[] data, subset;
    final int origLength;
    int index, attempts, copySource, checkIndex, destIndex;
    double x, y, inverseWeight;

    data = this.m_values;
    origLength = (data.length / 3);
    if ((npoints >= origLength) || (npoints <= 0)) {
      return this;
    }

    destIndex = (npoints * 3);
    subset = new double[destIndex];
    x = y = inverseWeight = Double.NaN;
    for (index = npoints; (--index) >= 0;) {
      attempter: for (attempts = 100; (--attempts) >= 0;) {
        copySource = (3 * (random.nextInt(origLength) + 1));
        x = data[--copySource];
        y = data[--copySource];
        inverseWeight = data[copySource - 1];
        for (checkIndex = (subset.length - 1); checkIndex > destIndex;) {
          if ((x == subset[checkIndex]) || //
              (y == subset[checkIndex - 1])) {
            continue attempter;
          }
          checkIndex -= 3;
        }
        break attempter;
      }

      subset[--destIndex] = x;
      subset[--destIndex] = y;
      subset[--destIndex] = inverseWeight;
    }

    return new WeightedRootMeanSquareError(subset);
  }

  /**
   * Compute the minimum inverse weight
   *
   * @param minY
   *          the minimal feasible Y value
   * @param minY2
   *          the second smallest feasible Y value
   * @return the minimum inverse weight
   */
  private static final double __getMinInverseWeight(final double minY,
      final double minY2) {
    double currentY;

    // define the weight of points with "0" y-coordinate
    if (WeightedRootMeanSquareError.__checkInverseWeight(minY)) {

      if (WeightedRootMeanSquareError.__checkInverseWeight(minY2)) {
        // Ideally, the inverse weight of the point with 0 y-value
        // behaves to the weight of the point with next-larger (i.e.,
        // smallest non-zero) absolute y value like this point's weight
        // to the one with second-smallest non-zero absolute y value.
        currentY = (minY * (minY / minY2));
        if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
          return currentY;
        }
      }

      // If that is not possible, we say it should just be ten percent
      // smaller.
      currentY = (minY * 0.9d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        return currentY;
      }

      // If that is not possible, we say it should just be one percent
      // smaller.
      currentY = (minY * 0.99d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        return currentY;
      }

      // If that is not possible, we say it should just be 0.1 percent
      // smaller.
      currentY = (minY * 0.999d);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        return currentY;
      }

      // If that is not possible, we say it should just be the tiniest
      // bit smaller.
      currentY = Math.nextAfter(minY, Double.NEGATIVE_INFINITY);
      if (WeightedRootMeanSquareError.__checkInverseWeight(currentY)) {
        return currentY;
      }

      // If that is not possible (smallest y-coordinate is
      // |Double.MIN_NORMAL|?), we say it should just be the same.
      return minY;
    }

    // If that is not possible (all weights are <=Double.MIN_NORMAL????),
    // we say it should just be 1.
    return 1d;
  }

  /**
   * Compute the matrix data
   *
   * @param matrix
   *          the matrix
   * @return the data array
   */
  private static final double[] __computeMatrix(final IMatrix matrix) {
    final double[] dest;
    final double minInverseWeight;
    int matrixIndex, dataIndex;
    double currentY, minY, minY2;

    FittingQualityMeasure.validateData(matrix);

    matrixIndex = matrix.m();
    dataIndex = (3 * matrixIndex);
    dest = new double[dataIndex];

    // find the two smallest non-zero absolute y values and copy the raw
    // data
    minY = minY2 = Double.POSITIVE_INFINITY;
    for (; (--matrixIndex) >= 0;) {
      dest[--dataIndex] = matrix.getDouble(matrixIndex, 0);
      dest[--dataIndex] = currentY = matrix.getDouble(matrixIndex, 1);
      if (currentY < 0d) {
        currentY = (-currentY);
      }
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
      --dataIndex;
    }

    minInverseWeight = WeightedRootMeanSquareError
        .__getMinInverseWeight(minY, minY2);
    for (dataIndex = dest.length; dataIndex > 0;) {
      currentY = dest[dataIndex -= 2];
      dest[--dataIndex] = Math.max(minInverseWeight, //
          Math.abs(currentY));
    }

    return dest;
  }

  /**
   * check the given inverse weight
   *
   * @param weight
   *          the weight
   * @return {@code true} if it can be used, {@code false} otherwise
   */
  private static final boolean __checkInverseWeight(final double weight) {
    return ((weight > Double.MIN_NORMAL)
        && (weight < Double.POSITIVE_INFINITY));
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "Weighted Root-Mean-Squared Error"; //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final int getSampleCount() {
    return (this.m_values.length / 3);
  }

}
