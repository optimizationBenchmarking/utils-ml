package org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex;

/** the class managing the solutions we have */
final class _CandidateManager {

  /** the visited candidates */
  private long[] m_done;

  /** the number of managed candidates */
  private int m_count;

  /**
   * create the candidate manager
   *
   * @param numParameters
   *          the number of parameters
   * @param initialCapacity
   *          the initial capacity
   */
  _CandidateManager(final int numParameters, final int initialCapacity) {
    super();
    this.m_done = new long[numParameters * initialCapacity];
  }

  /**
   * Add a given candidate to the managed list.
   *
   * @param candidate
   *          the candidate
   */
  final void _add(final _Candidate candidate) {
    final int numParams, start;
    long[] temp;

    numParams = candidate.m_bits.length;
    start = this.m_count * numParams;

    if (start >= this.m_done.length) {
      temp = new long[start * 2];
      System.arraycopy(this.m_done, 0, temp, 0, start);
      this.m_done = temp;
    }
    System.arraycopy(candidate.m_bits, 0, this.m_done, start, numParams);
    ++this.m_count;
  }

  /**
   * Check if a given candidate solution is sufficiently unique
   *
   * @param candidate
   *          the candidate solution
   * @param steps
   *          the number of steps required between at least two variables
   * @return {@code true} if there is no solution which has the same values
   *         in the {@code 64-bits} MSBs in each dimension, {@code false}
   *         otherwise
   */
  final boolean _isUniqueEnough(final _Candidate candidate,
      final long steps) {
    final int end, step;
    final long[] done;
    final long msteps;
    int mainIndex, innerIndex;

    step = candidate.m_bits.length;
    end = (this.m_count * step) - 1;
    done = this.m_done;
    msteps = (-steps);
    outer: for (mainIndex = (-1); mainIndex < end; mainIndex += step) {
      innerIndex = mainIndex;
      for (long value : candidate.m_bits) {
        value = (value - done[++innerIndex]);
        if ((value < msteps) || (value > steps)) {
          continue outer;
        }
        return false;
      }
    }
    return true;
  }
}
