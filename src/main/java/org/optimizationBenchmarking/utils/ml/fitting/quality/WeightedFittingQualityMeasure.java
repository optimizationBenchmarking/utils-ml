package org.optimizationBenchmarking.utils.ml.fitting.quality;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.StableSum;
import org.optimizationBenchmarking.utils.ml.fitting.spec.FittingEvaluation;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

/** The base class for weighted fitting quality measures */
public abstract class WeightedFittingQualityMeasure
    extends FittingQualityMeasure {

  /** the values backing this quality measure */
  private final double[] m_values;

  /** the sum to use for computing the fitting quality */
  private final StableSum m_sum;

  /**
   * create the fitting quality measure
   *
   * @param values
   *          the values array, of form
   *          {@code [inverseWeight1,y1,x1, inverseWeight2,y2,x2, ...]}
   */
  protected WeightedFittingQualityMeasure(final double[] values) {
    super();

    if ((values == null) || (values.length <= 0)
        || ((values.length % 3) != 0)) {
      throw new IllegalArgumentException(//
          "Invalid values array: must not be null or empty and must have a length which is a multiple of 3."); //$NON-NLS-1$
    }
    this.m_values = values;
    this.m_sum = new StableSum();
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final ParametricUnaryFunction model,
      final double[] parameters) {
    final StableSum sum;
    final double[] data;
    double residual;
    int index;

    sum = this.m_sum;
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
      final double[] parameters, final FittingEvaluation dest) {
    double[][] jacobian;
    double[] residuals;
    final int numSamples, numParams;
    final double[] data;
    final StableSum sum;
    double[] jacobianRow;
    double x, expectedY, residual, inverseWeight, squareErrorSum;
    int index, j;

    data = this.m_values;
    numSamples = (data.length / 3);

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

    for (index = data.length; index > 0;) {
      x = data[--index];
      expectedY = data[--index];
      inverseWeight = data[--index];

      residual = ((expectedY - model.value(x, parameters))
          / inverseWeight);

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

  /**
   * This internal method used for creating a compatible quality measure
   *
   * @param points
   *          the points
   * @return the measure
   */
  protected abstract WeightedFittingQualityMeasure create(
      final double[] points);

  /** {@inheritDoc} */
  @Override
  public final WeightedFittingQualityMeasure subselect(final int npoints,
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

    return this.create(subset);
  }
}
