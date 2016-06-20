package org.optimizationBenchmarking.utils.ml.clustering.impl.abstr;

/** The base class for data clustering jobs. */
public abstract class DataClusteringJob extends ClusteringJob {

  /**
   * create the clustering job
   *
   * @param builder
   *          the job builder
   */
  protected DataClusteringJob(final DataClusteringJobBuilder builder) {
    super(builder, false);
  }

  /** {@inheritDoc} **/
  @Override
  final ClusteringSolution _testTrivially() {
    final ClusteringSolution result;
    if (this.m_m == 2) {
      result = ClusteringTools.clusterTwoDataElements(this.getLogger(),
          this.m_matrix);
      this.m_matrix = null;
      return result;
    }
    return null;
  }
}
