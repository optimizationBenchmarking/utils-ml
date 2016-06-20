package org.optimizationBenchmarking.utils.ml.clustering.impl.abstr;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.hash.HashUtils;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringResult;

/** This record represents a solution to a clustering problem. */
public class ClusteringSolution implements IClusteringResult {

  /** the assignment of data rows to clusters */
  private final int[] m_assignment;

  /** the number of clusters */
  private final int m_count;

  /** the clustering quality */
  private final double m_quality;

  /**
   * Create the solution
   *
   * @param candidate
   *          the candidate to copy
   */
  ClusteringSolution(final ClusteringCandidateSolution candidate) {
    this(candidate.assignment, candidate.count, candidate.quality);
  }

  /**
   * Create the solution
   *
   * @param _assignment
   *          the assignment
   * @param _count
   *          the number of clusters
   * @param _quality
   *          the quality
   */
  ClusteringSolution(final int[] _assignment, final int _count,
      final double _quality) {
    super();
    this.m_assignment = _assignment;
    this.m_count = _count;
    this.m_quality = _quality;
  }

  /** {@inheritDoc} */
  @Override
  public final double getQuality() {
    return this.m_quality;
  }

  /** {@inheritDoc} */
  @Override
  public final int[] getClustersRef() {
    return this.m_assignment;
  }

  /** {@inheritDoc} */
  @Override
  public final int getClusterCount() {
    return this.m_count;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean equals(final Object o) {
    final ClusteringSolution s;
    if (o instanceof ClusteringSolution) {
      s = ((ClusteringSolution) o);
      return ((EComparison.EQUAL.compare(this.m_quality, s.m_quality))
          && (Arrays.equals(this.m_assignment, s.m_assignment)));
    }
    return false;
  }

  /** {@inheritDoc} */
  @Override
  public final int hashCode() {
    return HashUtils.combineHashes(//
        Arrays.hashCode(this.m_assignment), //
        HashUtils.hashCode(this.m_quality));
  }
}
