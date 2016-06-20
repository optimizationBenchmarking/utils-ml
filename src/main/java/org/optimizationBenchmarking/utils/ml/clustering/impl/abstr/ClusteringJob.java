package org.optimizationBenchmarking.utils.ml.clustering.impl.abstr;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringJob;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringResult;
import org.optimizationBenchmarking.utils.text.textOutput.MemoryTextOutput;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/** The base class for clustering jobs. */
public abstract class ClusteringJob extends ToolJob
    implements IClusteringJob {

  /** the minimum number of clusters */
  protected final int m_minClusters;
  /** the maximum number of clusters */
  protected final int m_maxClusters;

  /**
   * the number of rows of the matrix, i.e., the number of elements to
   * cluster
   */
  protected final int m_m;
  /** the number of columns */
  protected final int m_n;

  /** the matrix representing the data to be clustered */
  protected IMatrix m_matrix;

  /**
   * create the clustering job
   *
   * @param builder
   *          the job builder
   * @param isMatrixDistance
   *          is the matrix a distance matrix?
   */
  protected ClusteringJob(final ClusteringJobBuilder<?> builder,
      final boolean isMatrixDistance) {
    super(builder);

    int minClusters, maxClusters;

    ClusteringJobBuilder._checkClusterNumber(//
        minClusters = builder.m_minClusters, true);
    ClusteringJobBuilder._checkClusterNumber(//
        maxClusters = builder.m_maxClusters, true);

    if ((minClusters > 0) && (maxClusters > 0)
        && (minClusters > maxClusters)) {
      throw new IllegalArgumentException(//
          "The minimum number of clusters (" //$NON-NLS-1$
              + minClusters + //
              ") cannot be bigger than the maximum number of clusters ("//$NON-NLS-1$
              + maxClusters + "), but is."); //$NON-NLS-1$
    }

    ClusteringJobBuilder.checkMatrix(this.m_matrix = builder.m_matrix,
        isMatrixDistance);

    this.m_m = this.m_matrix.m();
    this.m_n = this.m_matrix.n();

    if (minClusters <= 0) {
      minClusters = this.computeMinClusters(maxClusters, this.m_m);
      ClusteringJobBuilder._checkClusterNumber(minClusters, false);
    }
    this.m_minClusters = minClusters;

    if (maxClusters <= 0) {
      maxClusters = this.computeMaxClusters(this.m_minClusters, this.m_m);
      ClusteringJobBuilder._checkClusterNumber(maxClusters, false);
    }
    this.m_maxClusters = maxClusters;

    if ((this.m_minClusters > this.m_maxClusters)) {
      throw new IllegalArgumentException((((((((((//
      "The minimum number of clusters (" //$NON-NLS-1$
          + this.m_minClusters) + //
          ") cannot be bigger than the maximum number of clusters (")//$NON-NLS-1$
          + this.m_maxClusters)
          + "), but based on your suggested minimum ") //$NON-NLS-1$
          + minClusters) + " and maximum ")//$NON-NLS-1$
          + maxClusters) + " and number of elements ")//$NON-NLS-1$
          + this.m_m) + ", we arrived there.");//$NON-NLS-1$
    }
  }

  /**
   * Compute the minimum number of clusters if it was not suggested
   *
   * @param maxClusters
   *          the suggested maximum number of clusters, or {@code -1} if
   *          unspecified
   * @param m
   *          the number of elements
   * @return the suggested minimum of clusters
   */
  protected int computeMinClusters(final int maxClusters, final int m) {
    return 1;
  }

  /**
   * Compute the maximum number of clusters if it was not suggested
   *
   * @param minClusters
   *          the suggested minimum number of clusters
   * @param m
   *          the number of elements
   * @return the suggested maximum of clusters
   */
  protected int computeMaxClusters(final int minClusters, final int m) {
    return Math.min(12, Math.max(minClusters, m));
  }

  /**
   * <p>
   * Perform the clustering and return a solution record.
   * </p>
   * <p>
   * The result of this method will be automatically
   * {@link ClusteringTools#normalizeClusters(int[])} normalized and the
   * number of clusters in it will be computed. In other words, you do not
   * need to set {@link ClusteringCandidateSolution#count} and the cluster
   * indexes do not need to be continuous or anything, i.e., you could even
   * return a cluster assignment like {@code 1, 1, 9, 9, 9, 4}, which would
   * then be transformed to {@code 1, 1, 0, 0, 0, 2}.
   * </p>
   *
   * @return the solution
   * @throws Exception
   *           if something goes wrong
   */
  protected abstract ClusteringCandidateSolution cluster()
      throws Exception;

  /**
   * Test whether the problem can be solved trivially
   *
   * @return the direct result, or {@code null} if the problem is
   *         non-trivial
   */
  abstract ClusteringSolution _testTrivially();

  /**
   * create the basic message body
   *
   * @return the message body
   */
  private final MemoryTextOutput __createMessageBody() {
    final MemoryTextOutput textOut;

    textOut = new MemoryTextOutput(512);
    textOut.append(' ');
    textOut.append('a');
    textOut.append(' ');
    textOut.append(this.m_m);
    textOut.append('x');
    textOut.append(this.m_n);
    textOut.append(" matrix into ");//$NON-NLS-1$
    if (this.m_minClusters >= this.m_maxClusters) {
      textOut.append(this.m_minClusters);
    } else {
      textOut.append('[');
      textOut.append(this.m_minClusters);
      textOut.append(',');
      textOut.append(this.m_maxClusters);
      textOut.append(']');
    }
    textOut.append(" clusters with method ");//$NON-NLS-1$
    textOut.append(this.toString());
    if (this instanceof DataClusteringJob) {
      textOut.append(" (a raw data clustering method)");//$NON-NLS-1$
    } else {
      if (this instanceof DistanceClusteringJob) {
        textOut.append(" (a distance-based clustering method)");//$NON-NLS-1$
      }
    }
    return textOut;
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return this.getClass().getSimpleName();
  }

  /** {@inheritDoc} */
  @SuppressWarnings("null")
  @Override
  public final IClusteringResult call() throws IllegalArgumentException {
    final Logger logger;
    MemoryTextOutput textOut;
    ClusteringCandidateSolution solution;
    Throwable error;
    ClusteringSolution trivial;
    String message;
    char separator;
    boolean canLog, isFinite, wrongNumber;

    textOut = null;
    message = null;
    error = null;
    try {
      logger = this.getLogger();
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        textOut = this.__createMessageBody();
        message = textOut.toString();
        logger.finer("Beginning to cluster" + message);//$NON-NLS-1$
      }

      try {
        trivial = this._testTrivially();
        if (trivial != null) {
          return trivial;
        }

        solution = this.cluster();

        isFinite = MathUtils.isFinite(solution.quality);
        if (isFinite) {
          solution.count = ClusteringTools
              .normalizeClusters(solution.assignment);
          wrongNumber = ((solution.count > this.m_maxClusters)
              || (solution.count < this.m_minClusters));
        } else {
          solution.count = (-1);
          wrongNumber = true;
        }

        canLog = (logger != null) && (logger.isLoggable(Level.FINER));

        if (canLog || (!isFinite) || wrongNumber) {
          if (textOut == null) {
            textOut = this.__createMessageBody();
          }
          textOut.append(", obtained assignment ");//$NON-NLS-1$
          separator = '[';
          for (final int assignment : solution.assignment) {
            textOut.append(separator);
            separator = ',';
            textOut.append(assignment);
          }
          textOut.append("] with quality ");//$NON-NLS-1$
          textOut.append(solution.quality);
          if (solution.count >= 0) {
            textOut.append(" corresponding to ");//$NON-NLS-1$
            textOut.append(solution.count);
            textOut.append(" clusters");//$NON-NLS-1$
          }
          message = null;
        }

        if (isFinite && (!wrongNumber)) {
          if (canLog) {
            textOut.append('.');
            logger.finer("Finished clustering" + //$NON-NLS-1$
                textOut.toString());
          }
          return new ClusteringSolution(solution);
        }
      } catch (final Throwable cause) {
        error = cause;
      } finally {
        this.m_matrix = null;
      }

      if (textOut == null) {
        textOut = this.__createMessageBody();
      }
      if (message == null) {
        message = textOut.toString();
      }
      message = ("Error while trying to cluster" + message + '.'); //$NON-NLS-1$
      if (error != null) {
        throw new IllegalArgumentException(message, error);
      }
      throw new IllegalArgumentException(message);
    } finally {
      this.m_matrix = null;
    }
  }
}
