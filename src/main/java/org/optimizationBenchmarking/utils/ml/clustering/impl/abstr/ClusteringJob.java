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

  /** the number of classes, {@code -1} if unspecified */
  protected final int m_classes;

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

    ClusteringJobBuilder._checkClusterNumber(//
        this.m_classes = builder.m_classes, true);
    ClusteringJobBuilder.checkMatrix(this.m_matrix = builder.m_matrix,
        isMatrixDistance);
  }

  /**
   * Perform the clustering and return a solution record. The result of
   * this method will be automatically
   * {@link ClusteringTools#normalizeClusters(int[])} normalized.
   *
   * @return the solution
   * @throws Exception
   *           if something goes wrong
   */
  ClusteringSolution _cluster() throws Exception {
    return this.cluster();
  }

  /**
   * Perform the clustering and return a solution record. The result of
   * this method will be automatically
   * {@link ClusteringTools#normalizeClusters(int[])} normalized.
   *
   * @return the solution
   * @throws Exception
   *           if something goes wrong
   */
  protected abstract ClusteringSolution cluster() throws Exception;

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
    textOut.append(this.m_matrix.m());
    textOut.append('x');
    textOut.append(this.m_matrix.n());
    if (this.m_classes > 0) {
      textOut.append(" matrix into ");//$NON-NLS-1$
      textOut.append(this.m_classes);
    } else {
      textOut.append(" matrix an arbitrary number of");//$NON-NLS-1$
    }
    textOut.append(" classes with method ");//$NON-NLS-1$
    textOut.append(this.getClass().getSimpleName());
    textOut.append(this.getClass().getSimpleName());
    if (this instanceof DataClusteringJob) {
      textOut.append(" (a raw data clustering method)");//$NON-NLS-1$
    } else {
      textOut.append(this.getClass().getSimpleName());
      if (this instanceof DistanceClusteringJob) {
        textOut.append(" (a distance-based clustering method)");//$NON-NLS-1$
      }
    }
    return textOut;
  }

  /** {@inheritDoc} */
  @SuppressWarnings("null")
  @Override
  public final IClusteringResult call() throws IllegalArgumentException {
    final Logger logger;
    MemoryTextOutput textOut;
    ClusteringSolution solution;
    Throwable error;
    String message;
    char separator;
    boolean canLog, isFinite;

    textOut = null;
    message = null;
    error = null;

    logger = this.getLogger();
    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      textOut = this.__createMessageBody();
      message = textOut.toString();
      logger.finer("Beginning to cluster" + message);//$NON-NLS-1$
    }

    try {
      solution = this._cluster();

      isFinite = MathUtils.isFinite(solution.quality);
      if (isFinite) {
        ClusteringTools.normalizeClusters(solution.assignment);
      }

      canLog = (logger != null) && (logger.isLoggable(Level.FINER));

      if (canLog || (!isFinite)) {
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
        message = null;
      }

      if (isFinite) {
        if (canLog) {
          textOut.append('.');
          logger.finer("Finished clustering" + //$NON-NLS-1$
              textOut.toString());
        }
        return solution;
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
  }
}
