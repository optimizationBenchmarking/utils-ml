package org.optimizationBenchmarking.utils.ml.fitting.impl.esls;

import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingCandidateSolution;

/** the candidate solution as used in a evolution strategy */
final class _ESCandidate extends FittingCandidateSolution {

  /** the step length */
  final double[] m_stepLength;

  /**
   * create
   *
   * @param numParams
   *          the number of parameters
   */
  _ESCandidate(final int numParams) {
    super(numParams);
    this.m_stepLength = new double[numParams];
  }
}
