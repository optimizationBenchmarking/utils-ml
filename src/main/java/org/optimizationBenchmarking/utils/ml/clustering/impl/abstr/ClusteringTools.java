package org.optimizationBenchmarking.utils.ml.clustering.impl.abstr;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.comparison.Compare;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.combinatorics.CanonicalPermutation;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;

/** Tools for clustering */
public final class ClusteringTools {

  /** the forbidden constructor */
  private ClusteringTools() {
    ErrorUtils.doNotCall();
  }

  /**
   * check if two {@code double} values are sufficiently equal so that they
   * can be ignored.
   *
   * @param a
   *          the first {@code double} value
   * @param b
   *          the second {@code double} value
   * @return {@code true} if both are sufficiently equal
   */
  private static final boolean __equals(final double a, final double b) {
    return (MathUtils.numbersBetween(a, b) <= 3);
  }

  /**
   * Preprocess a given matrix: delete useless columns, normalize columns
   *
   * @param matrix
   *          the matrix
   * @return a preprocessed version
   */
  public static final DoubleMatrix1D preprocessDataMatrix(
      final IMatrix matrix) {
    final double[] min, max;
    final int[] columns;
    final int m;
    double[] data, data2;
    int realN, n, i, j, rj, k;
    double d;

    m = matrix.m();
    n = matrix.n();
    min = new double[n];
    Arrays.fill(min, Double.POSITIVE_INFINITY);
    max = new double[n];
    Arrays.fill(max, Double.NEGATIVE_INFINITY);

    // find minima and maxima
    for (i = m; (--i) >= 0;) {
      for (j = n; (--j) >= 0;) {
        d = matrix.getDouble(i, j);
        if (d < min[j]) {
          min[j] = d;
        }
        if (d > max[j]) {
          max[j] = d;
        }
      }
    }

    // allocate the column list
    columns = new int[n];
    for (j = n; (--j) >= 0;) {
      columns[j] = j;
    }
    realN = n;

    // check for columns that can be deleted based on their raw data values
    outer: for (j = n; (--j) >= 0;) {
      if (ClusteringTools.__equals(min[j], max[j])) {
        // the column contains only one single value
        columns[j] = columns[--realN];
        continue outer;
      }

      // check if the column equals another column
      inner: for (i = j; (--i) >= 0;) {
        if (ClusteringTools.__equals(min[j], min[i])
            && ClusteringTools.__equals(max[j], max[i])) {
          // well, there minimum and maximum are the same, so maybe...
          for (k = m; (--k) >= 0;) {
            if (!(ClusteringTools.__equals(matrix.getDouble(k, j),
                matrix.getDouble(k, i)))) {
              continue inner;
            }
          }
          // ok, they are similar
          columns[j] = columns[--realN];
          continue outer;
        }
      }
    }

    // now normalize the data
    for (j = n; (--j) >= 0;) {
      max[j] -= min[j];
    }

    k = (m * realN);
    data = new double[k];
    for (i = m; (--i) >= 0;) {
      for (j = realN; (--j) >= 0;) {
        rj = columns[j];
        data[--k] = Math.min(1d, Math.max(0d, //
            ((matrix.getDouble(i, rj) - min[rj]) / max[rj])));
      }
    }

    // now check again if some columns are redundant
    for (j = realN; (--j) >= 0;) {
      columns[j] = j;
    }

    n = realN;
    outer2: for (j = realN; (--j) > 0;) {

      inner2: for (i = j; (--i) >= 0;) {

        for (k = ((m - 1) * n); k >= 0; k -= n) {
          if (!(ClusteringTools.__equals(data[k + j], data[k + i]))) {
            continue inner2;
          }
        }

        // ok, they are similar
        columns[j] = columns[--realN];
        continue outer2;
      }
    }

    // Did we delete some data now?
    if (realN != n) {
      // Great, let's re-allocate
      k = (realN * m);
      data2 = new double[k];
      for (i = m; (--i) >= 0;) {
        for (j = realN; (--j) >= 0;) {
          data2[(i * realN) + j] = data[(i * n) + columns[j]];
        }
      }
      data = data2;
      n = realN;
    }

    return new DoubleMatrix1D(data, m, n);
  }

  /**
   * Check whether there is a simple, default solution for the clustering
   * problem
   *
   * @param logger
   *          the logger to use
   * @param matrix
   *          the data or dissimilarity matrix
   * @param minClusters
   *          the minimum number of classes we want
   * @param maxClusters
   *          the maximum number of classes we want
   * @return the default solution, or {@code null} if there is none
   * @throws IllegalArgumentException
   *           if the clustering job cannot be done
   */
  public static final ClusteringSolution canClusterTrivially(
      final Logger logger, final IMatrix matrix, final int minClusters,
      final int maxClusters) {
    return ClusteringTools._canClusterTrivially(logger, matrix,
        minClusters, maxClusters);
  }

  /**
   * Check whether there is a simple, default solution for the clustering
   * problem
   *
   * @param logger
   *          the logger to use
   * @param matrix
   *          the data or dissimilarity matrix
   * @param minClusters
   *          the minimum number of classes we want
   * @param maxClusters
   *          the maximum number of classes we want
   * @return the default solution, or {@code null} if there is none
   * @throws IllegalArgumentException
   *           if the clustering job cannot be done
   */
  static final _DirectResult _canClusterTrivially(final Logger logger,
      final IMatrix matrix, final int minClusters, final int maxClusters) {
    final int m;

    m = matrix.m();
    if (minClusters > m) {
      throw new IllegalArgumentException(
          m + " data samples cannot be divided into " //$NON-NLS-1$
              + minClusters + //
              " clusters."); //$NON-NLS-1$
    }

    if (m <= minClusters) {
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        logger.fine("The minimum number of clusters to use is " //$NON-NLS-1$
            + minClusters + " and there are " + m + //$NON-NLS-1$
            " data elements to cluster, so we put each in one cluster."); //$NON-NLS-1$

      }
      return new _DirectResult(CanonicalPermutation.createCanonicalZero(m),
          m, 0d);
    }

    if (m == 1) {
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        logger.fine(//
            "There is only one element to cluster, so we put it into its own, single cluster."); //$NON-NLS-1$

      }
      return new _DirectResult(new int[1], 1, 0d);
    }

    if ((maxClusters >= 0) && (maxClusters <= 1)) {
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        logger.fine(//
            "The maximum number of clusters is one, so we put all elements into a single cluster."); //$NON-NLS-1$

      }
      return new _DirectResult(new int[m], 1, 0d);
    }

    return null;
  }

  /**
   * Cluster two data elements: If we have two
   *
   * @param logger
   *          the logger to use
   * @param matrix
   *          the data or dissimilarity matrix
   * @return the default solution, or {@code null} if there is none
   */
  public static final ClusteringSolution clusterTwoDataElements(
      final Logger logger, final IMatrix matrix) {
    final _DirectResult result;
    final String message;
    int index;

    isTwoClusters: {
      isSingleCluster: {
        if (matrix.isIntegerMatrix()) {
          for (index = matrix.n(); (--index) >= 0;) {
            if (matrix.getLong(0, index) != matrix.getLong(1, index)) {
              break isSingleCluster;
            }
          }
        } else {
          for (index = matrix.n(); (--index) >= 0;) {
            if (!(Compare.equals(matrix.getDouble(0, index),
                matrix.getDouble(1, index)))) {
              break isSingleCluster;
            }
          }
        }
        result = new _DirectResult(new int[2], 1, 0d);
        message = "The data set contains exactly two identical data samples. We will put them into one single cluster.";//$NON-NLS-1$
        break isTwoClusters;
      }
      result = new _DirectResult(new int[] { 0, 1 }, 2, 0d);
      message = "The data set contains exactly two different data samples. We will put each one into a different cluster, i.e., get two clusters."; //$NON-NLS-1$
    }

    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      logger.finer(message);
    }
    return result;
  }

  /**
   * Cluster two distance elements: If we have two
   *
   * @param logger
   *          the logger to use
   * @param matrix
   *          the data or dissimilarity matrix
   * @return the default solution, or {@code null} if there is none
   */
  public static final ClusteringSolution clusterTwoDistanceElements(
      final Logger logger, final IMatrix matrix) {
    final _DirectResult result;
    final String message;

    isTwoClusters: {
      isSingleCluster: {
        if (matrix.isIntegerMatrix()) {
          if (matrix.getLong(0, 1) != 0L) {
            break isSingleCluster;
          }
        } else {
          if (matrix.getDouble(0, 1) != 0d) {
            break isSingleCluster;
          }
        }
        result = new _DirectResult(new int[2], 1, 0d);
        message = "The data set contains exactly two samples with distance 0, i.e., two identical samples. We will put them into one single cluster.";//$NON-NLS-1$
        break isTwoClusters;
      }
      result = new _DirectResult(new int[] { 0, 1 }, 2, 0d);
      message = "The data set contains exactly two data samples with non-zero distance, i.e., two different samples. We will put each one into a different cluster, i.e., get two clusters."; //$NON-NLS-1$
    }

    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      logger.finer(message);
    }
    return result;
  }

  /**
   * Normalize the clusters: We sort clusters by their size, bigger
   * clusters come first. Clusters of equal size are sorted according to
   * their smallest member index.
   *
   * @param clusters
   *          the cluster array to normalize
   * @return the number of clusters
   */
  public static final int normalizeClusters(final int[] clusters) {
    final __TempCluster[] alloc;
    __TempCluster[] sorted;
    __TempCluster cur;
    int min, max, i;

    min = Integer.MAX_VALUE;
    max = Integer.MIN_VALUE;

    for (final int a : clusters) {
      if (a < min) {
        min = a;
      }
      if (a > max) {
        max = a;
      }
    }

    if (min >= max) {
      Arrays.fill(clusters, 0);
      return 1;
    }

    alloc = new __TempCluster[(max - min) + 1];
    for (i = alloc.length; (--i) >= 0;) {
      alloc[i] = new __TempCluster();
    }
    i = 0;
    for (final int a : clusters) {
      cur = alloc[a - min];
      ++cur.m_size;
      ++i;
      if (i < cur.m_minMember) {
        cur.m_minMember = i;
      }
    }

    sorted = alloc.clone();
    Arrays.sort(sorted);
    for (i = sorted.length; (--i) >= 0;) {
      sorted[i].m_newID = i;
    }
    sorted = null;

    i = (-1);
    for (final int a : clusters) {
      clusters[++i] = alloc[a - min].m_newID;
    }

    return alloc.length;
  }

  /** a temporary cluster */
  private static final class __TempCluster
      implements Comparable<__TempCluster> {

    /** the members */
    int m_minMember;

    /** the number of members */
    int m_size;

    /** the new cluster id */
    int m_newID;

    /** create */
    __TempCluster() {
      super();
      this.m_minMember = Integer.MAX_VALUE;
    }

    /** {@inheritDoc} */
    @Override
    public final int compareTo(final __TempCluster o) {
      int res;

      res = Integer.compare(o.m_size, this.m_size);
      if (res != 0) {
        return res;
      }

      return Integer.compare(this.m_minMember, o.m_minMember);
    }
  }
}
