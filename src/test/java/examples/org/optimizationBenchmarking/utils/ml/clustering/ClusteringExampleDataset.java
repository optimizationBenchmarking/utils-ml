package examples.org.optimizationBenchmarking.utils.ml.clustering;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;

/** a clustering example */
public final class ClusteringExampleDataset
    implements Comparable<ClusteringExampleDataset> {

  /** the name of the example */
  public final String name;

  /** the matrix */
  public final DoubleMatrix1D data;

  /** the number of anticipated clusters */
  public final int classes;

  /** the goal clusters */
  public final int[] clusters;

  /**
   * create the clustering example
   *
   * @param _name
   *          the example name
   * @param _data
   *          the data
   * @param _classes
   *          the goal clusters
   * @param _clusters
   *          the clusters
   */
  private ClusteringExampleDataset(final String _name,
      final DoubleMatrix1D _data, final int _classes,
      final int[] _clusters) {
    super();
    this.data = _data;
    this.name = _name;
    this.classes = _classes;
    this.clusters = _clusters;
  }

  /**
   * load an example data set
   *
   * @param name
   *          the data set name
   * @return the data set
   * @throws IOException
   *           if i/o fails
   */
  static final ClusteringExampleDataset _load(final String name)
      throws IOException {
    final ArrayList<double[]> features;
    final ArrayList<Integer> classes;
    final int n, m;
    String[] featuresStr;
    double[] cur;
    int[] assignments;
    int i;
    String s;

    try (final InputStream is = ClusteringExampleDatasets.class
        .getResourceAsStream(//
            name + ".txt")) {//$NON-NLS-1$
      try (final InputStreamReader isr = new InputStreamReader(is)) {
        try (final BufferedReader br = new BufferedReader(isr)) {
          features = new ArrayList<>();
          classes = new ArrayList<>();

          while ((s = br.readLine()) != null) {
            s = s.trim();
            if (s.length() <= 0) {
              continue;
            }

            featuresStr = s.split(","); //$NON-NLS-1$
            i = featuresStr.length;

            classes.add(//
                Integer.valueOf((featuresStr[--i].charAt(0) - 'A')));
            cur = new double[i];
            for (; (--i) >= 0;) {
              cur[i] = Double.parseDouble(featuresStr[i]);
            }
            features.add(cur);
          }
        }
      }
    }

    m = classes.size();
    assignments = new int[m];
    for (i = m; (--i) >= 0;) {
      assignments[i] = classes.get(i).intValue();
    }

    n = features.get(0).length;
    cur = new double[m * n];
    i = 0;
    for (final double[] vec : features) {
      System.arraycopy(vec, 0, cur, i, n);
      i += n;
    }

    return new ClusteringExampleDataset(name,
        new DoubleMatrix1D(cur, m, n), assignments[m - 1] + 1,
        assignments);
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final ClusteringExampleDataset o) {
    return String.CASE_INSENSITIVE_ORDER.compare(this.name, o.name);
  }
}
