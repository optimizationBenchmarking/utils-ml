package examples.org.optimizationBenchmarking.utils.ml.clustering;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ClusteringExampleDataset;

/** The loader for the clustering example data sets. */
public final class ClusteringExampleDatasets {

  /** the simple example 2 */
  public static final ClusteringExampleDataset SIMPLE_2 = //
  ClusteringExampleDatasets.__load("simple2"); //$NON-NLS-1$
  /** the simple example 3 */
  public static final ClusteringExampleDataset SIMPLE_3 = //
  ClusteringExampleDatasets.__load("simple3"); //$NON-NLS-1$
  /** the simple example 4 */
  public static final ClusteringExampleDataset SIMPLE_4 = //
  ClusteringExampleDatasets.__load("simple4"); //$NON-NLS-1$
  /** the exponential example 3/2 */
  public static final ClusteringExampleDataset EXP_3_2 = //
  ClusteringExampleDatasets.__load("exp3-2"); //$NON-NLS-1$
  /** the exponential example 4/2 */
  public static final ClusteringExampleDataset EXP_4_2 = //
  ClusteringExampleDatasets.__load("exp4-2"); //$NON-NLS-1$
  /** the exponential example 5/2 */
  public static final ClusteringExampleDataset EXP_5_2 = //
  ClusteringExampleDatasets.__load("exp5-2"); //$NON-NLS-1$
  /** the exponential example 6/2 */
  public static final ClusteringExampleDataset EXP_6_2 = //
  ClusteringExampleDatasets.__load("exp6-2"); //$NON-NLS-1$
  /** the exponential example 3/10 */
  public static final ClusteringExampleDataset EXP_3_10 = //
  ClusteringExampleDatasets.__load("exp3-10"); //$NON-NLS-1$
  /** the exponential example 4/10 */
  public static final ClusteringExampleDataset EXP_4_10 = //
  ClusteringExampleDatasets.__load("exp4-10"); //$NON-NLS-1$
  /** the exponential example 5/10 */
  public static final ClusteringExampleDataset EXP_5_10 = //
  ClusteringExampleDatasets.__load("exp5-10"); //$NON-NLS-1$
  /** the iris example */
  public static final ClusteringExampleDataset IRIS = //
  ClusteringExampleDatasets.__load("iris"); //$NON-NLS-1$

  /** the examples */
  public static final ArrayListView<ClusteringExampleDataset> EXAMPLES;

  static {
    final ClusteringExampleDataset[] examples;

    examples = new ClusteringExampleDataset[] {
        ClusteringExampleDatasets.SIMPLE_2,
        ClusteringExampleDatasets.SIMPLE_3,
        ClusteringExampleDatasets.SIMPLE_4, //
        ClusteringExampleDatasets.EXP_3_2,
        ClusteringExampleDatasets.EXP_4_2,
        ClusteringExampleDatasets.EXP_5_2,
        ClusteringExampleDatasets.EXP_6_2, //
        ClusteringExampleDatasets.EXP_3_10,
        ClusteringExampleDatasets.EXP_4_10,
        ClusteringExampleDatasets.EXP_5_10, //
        ClusteringExampleDatasets.IRIS//
    };
    Arrays.sort(examples);

    EXAMPLES = new ArrayListView<>(examples, false);
  }

  /** create */
  private ClusteringExampleDatasets() {
    ErrorUtils.doNotCall();
  }

  /**
   * load an example data set
   *
   * @param name
   *          the data set name
   * @return the data set
   */
  private static final ClusteringExampleDataset __load(final String name) {
    final ArrayList<double[]> features;
    final ArrayList<Integer> classes;
    final HashSet<Integer> classSet;
    final int n, m;
    String[] featuresStr;
    double[] cur;
    int[] assignments;
    int i;
    String s;
    Integer clazz;

    try (final InputStream is = ClusteringExampleDatasets.class
        .getResourceAsStream(//
            name + ".txt")) {//$NON-NLS-1$
      try (final InputStreamReader isr = new InputStreamReader(is)) {
        try (final BufferedReader br = new BufferedReader(isr)) {
          features = new ArrayList<>();
          classes = new ArrayList<>();
          classSet = new HashSet<>();

          while ((s = br.readLine()) != null) {
            s = s.trim();
            if (s.length() <= 0) {
              continue;
            }

            featuresStr = s.split(","); //$NON-NLS-1$
            i = featuresStr.length;

            clazz = Integer.valueOf((featuresStr[--i].charAt(0) - 'A'));
            classes.add(clazz);
            classSet.add(clazz);
            cur = new double[i];
            for (; (--i) >= 0;) {
              cur[i] = Double.parseDouble(featuresStr[i]);
            }
            features.add(cur);
          }
        }
      }
    } catch (final Throwable error) {
      throw new RuntimeException(error);
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
        new DoubleMatrix1D(cur, m, n), classSet.size(), assignments);
  }
}
