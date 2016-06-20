package org.optimizationBenchmarking.utils.ml.clustering.impl.abstr;

import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.EuclideanDistance;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.MeasureBasedDistanceMatrixBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceMeasure;

/** The base class for distance-based clustering jobs. */
public abstract class DistanceClusteringJob extends ClusteringJob {
  /**
   * is the matrix a distance matrix ({@code true}) or a data matrix (
   * {@code false})?
   */
  private final boolean m_matrixIsDistanceMatrix;

  /**
   * create the clustering job
   *
   * @param builder
   *          the job builder
   */
  protected DistanceClusteringJob(
      final DistanceClusteringJobBuilder builder) {
    super(builder, builder.m_matrixIsDistanceMatrix);
    this.m_matrixIsDistanceMatrix = builder.m_matrixIsDistanceMatrix;
  }

  /**
   * Create the distance measure to be used for converting data matrices to
   * distance matrices.
   *
   * @return the distance measure
   */
  protected IDistanceMeasure createDistanceMeasure() {
    return new EuclideanDistance();
  }

  /** {@inheritDoc} **/
  @Override
  final ClusteringSolution _testTrivially() {
    final Logger logger;
    final ClusteringSolution result;

    if (this.m_m == 2) {
      logger = this.getLogger();
      if (this.m_matrixIsDistanceMatrix) {
        result = ClusteringTools.clusterTwoDataElements(logger,
            this.m_matrix);
      } else {
        result = ClusteringTools.clusterTwoDistanceElements(logger,
            this.m_matrix);
      }
      this.m_matrix = null;
      return result;
    }

    if (!(this.m_matrixIsDistanceMatrix)) {
      this.m_matrix = new MeasureBasedDistanceMatrixBuilder(this.m_matrix,
          this.createDistanceMeasure()).call();
    }
    return null;
  }
}
