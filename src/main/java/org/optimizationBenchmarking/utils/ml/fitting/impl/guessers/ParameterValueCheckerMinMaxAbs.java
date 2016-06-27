package org.optimizationBenchmarking.utils.ml.fitting.impl.guessers;

/** A parameter value checker with a minimum and maximum absolute value */
public final class ParameterValueCheckerMinMaxAbs
    extends ParameterValueChecker {

  /** the minimum permitted absolute value */
  private final double m_minAbs;
  /** the maximum permitted absolute value */
  private final double m_maxAbs;

  /**
   * create
   *
   * @param minAbs
   *          the minimum absolute value
   * @param maxAbs
   *          the maximum absolute value
   */
  public ParameterValueCheckerMinMaxAbs(final double minAbs,
      final double maxAbs) {
    super();

    if (minAbs <= 0d) {
      throw new IllegalArgumentException(//
          "Minimum absolute value must be positive, but is " + //$NON-NLS-1$
              maxAbs);
    }
    if (maxAbs <= 0d) {
      throw new IllegalArgumentException(//
          "Maximum absolute value must be positive, but is " + //$NON-NLS-1$
              maxAbs);
    }
    if (minAbs >= maxAbs) {
      throw new IllegalArgumentException(//
          "Minimum absolute value (" + minAbs + //$NON-NLS-1$
              ") must be smaller than maximum absolute value (" //$NON-NLS-1$
              + maxAbs + "), but is not."); //$NON-NLS-1$
    }
    this.m_minAbs = minAbs;
    this.m_maxAbs = maxAbs;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean check(final double value) {
    final double abs;
    if (value == value) {
      abs = Math.abs(value);
      return ((abs >= this.m_minAbs) && (abs <= this.m_maxAbs));
    }
    return false;
  }
}
