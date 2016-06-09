package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingCandidateSolution;

/** an internal candidate solution */
final class _Candidate extends FittingCandidateSolution {

  /** the data */
  final long[] m_bits;

  /**
   * Create the fitting candidate solution
   *
   * @param parameterCount
   *          the number of parameters
   */
  _Candidate(final int parameterCount) {
    super(parameterCount);
    this.m_bits = new long[parameterCount];
  }

  /** {@inheritDoc} */
  @Override
  public final void assign(final double[] _solution,
      final double _quality) {
    final double[] dblDest;
    final long[] bitDest;
    int index;

    this.quality = _quality;
    dblDest = this.solution;
    bitDest = this.m_bits;
    index = (-1);

    for (final double value : _solution) {
      ++index;
      bitDest[index] = Double.doubleToLongBits(//
          dblDest[index] = (value + 0d));
    }
  }

  /**
   * copy an existing record
   *
   * @param other
   *          the record to copy
   */
  final void _assign(final _Candidate other) {
    System.arraycopy(other.solution, 0, this.solution, 0,
        this.solution.length);
    System.arraycopy(other.m_bits, 0, this.m_bits, 0, this.m_bits.length);
    this.quality = other.quality;
  }
}
