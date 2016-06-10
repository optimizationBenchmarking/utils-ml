package org.optimizationBenchmarking.utils.ml.fitting.quality;

import org.optimizationBenchmarking.utils.math.matrix.IMatrix;

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
    extends WeightedFittingQualityMeasure {
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
    super(data);
  }

  /** {@inheritDoc} */
  @Override
  protected final WeightedRootMeanSquareError create(final double[] data) {
    return new WeightedRootMeanSquareError(data);
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
}
