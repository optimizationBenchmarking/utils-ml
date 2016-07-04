package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import org.optimizationBenchmarking.utils.hash.HashUtils;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingResult;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

/** A fitting result record. */
public final class FittingResult implements IFittingResult {

  /** the unary function */
  private final ParametricUnaryFunction m_function;

  /** the fitting solution */
  private final double[] m_solution;

  /** the solution quality */
  private final double m_quality;

  /** the internal hash code */
  private int m_hashCode;

  /**
   * create the optimization result
   *
   * @param result
   *          the result array
   * @param quality
   *          the quality
   * @param function
   *          the function
   */
  FittingResult(final double[] result, final double quality,
      final ParametricUnaryFunction function) {
    this.m_function = function;
    this.m_quality = quality;
    this.m_solution = result;
  }

  /** {@inheritDoc} */
  @Override
  public final double getQuality() {
    return this.m_quality;
  }

  /** {@inheritDoc} */
  @Override
  public final double[] getFittedParametersRef() {
    return this.m_solution;
  }

  /** {@inheritDoc} */
  @Override
  public final ParametricUnaryFunction getFittedFunction() {
    return this.m_function;
  }

  /** {@inheritDoc} */
  @Override
  public final int hashCode() {
    if (this.m_hashCode == 0) {
      this.m_hashCode = HashUtils.combineHashes(//
          HashUtils.hashCode(this.m_function), //
          super.hashCode());
      if (this.m_hashCode == 0) {
        this.m_hashCode = FittingResult.class.hashCode();
      }
    }
    return this.m_hashCode;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean equals(final Object o) {
    final FittingResult other;

    if (o == this) {
      return true;
    }

    if (o instanceof FittingResult) {
      other = ((FittingResult) o);
      return ((other.m_function == this.m_function) && //
          super.equals(o));
    }
    return super.equals(o);
  }
}
