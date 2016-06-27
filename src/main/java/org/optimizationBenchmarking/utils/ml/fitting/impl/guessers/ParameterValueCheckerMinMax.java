package org.optimizationBenchmarking.utils.ml.fitting.impl.guessers;

/** A parameter value checker with a minimum and maximum value */
public final class ParameterValueCheckerMinMax
    extends ParameterValueChecker {

  /** the minimum permitted value */
  private final double m_min;
  /** the maximum permitted value */
  private final double m_max;

  /**
   * create
   *
   * @param min
   *          the minimum absolute value
   * @param max
   *          the maximum absolute value
   */
  public ParameterValueCheckerMinMax(final double min, final double max) {
    super();

    if (min >= max) {
      throw new IllegalArgumentException(//
          "Minimum value (" + min + //$NON-NLS-1$
              ") must be smaller than maximum value (" //$NON-NLS-1$
              + max + "), but is not."); //$NON-NLS-1$
    }
    this.m_min = min;
    this.m_max = max;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean check(final double value) {
    return ((value == value) && (value >= this.m_min)
        && (value <= this.m_max));
  }
}
