package shared.junit.org.optimizationBenchmarking.utils.ml.clustering;

import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;
import org.optimizationBenchmarking.utils.math.matrix.AbstractMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.impl.IntMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.impl.LongMatrix1D;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringJobBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusteringJobBuilder;

import examples.org.optimizationBenchmarking.utils.ml.clustering.ClusteringExampleDatasets;

/** The test for data clustering. */
@Ignore
public abstract class ExampleDataClustererTest extends DataClustererTest {

  /**
   * create
   *
   * @param clusterer
   *          the clusterer
   */
  public ExampleDataClustererTest(final IDataClusterer clusterer) {
    super(clusterer);
  }

  /**
   * test whether clustering the IRIS data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testIRISwithNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.IRIS, true);
  }

  /**
   * test whether clustering the IRIS data set without using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testIRISwithoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.IRIS, false);
  }

  /**
   * test whether clustering the Simple-2 data set with using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple2withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_2, true);
  }

  /**
   * test whether clustering the Simple-2 data set without using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple2withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_2, false);
  }

  /**
   * test whether clustering the Simple-3 data set with using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple3withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_3, true);
  }

  /**
   * test whether clustering the Simple-3 data set without using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple3withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_3, false);
  }

  /**
   * test whether clustering the Simple-4 data set with using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple4withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_4, true);
  }

  /**
   * test whether clustering the Simple-4 data set without using the
   * clustering number works
   */
  @Test(timeout = 3600000)
  public void testSimple4withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.SIMPLE_4, false);
  }

  /**
   * test whether clustering the Exp3-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp32withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_3_2, true);
  }

  /**
   * test whether clustering the Exp3-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp32withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_3_2, false);
  }

  /**
   * test whether clustering the Exp4-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp42withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_4_2, true);
  }

  /**
   * test whether clustering the Exp4-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp42withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_4_2, false);
  }

  /**
   * test whether clustering the Exp5-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp52withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_5_2, true);
  }

  /**
   * test whether clustering the Exp5-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp52withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_5_2, false);
  }

  /**
   * test whether clustering the Exp6-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp62withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_6_2, true);
  }

  /**
   * test whether clustering the Exp6-2 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp62withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_6_2, false);
  }

  /**
   * test whether clustering the Exp3-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp310withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_3_10, true);
  }

  /**
   * test whether clustering the Exp3-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp310withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_3_10, false);
  }

  /**
   * test whether clustering the Exp4-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp410withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_4_10, true);
  }

  /**
   * test whether clustering the Exp4-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp410withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_4_10, false);
  }

  /**
   * test whether clustering the Exp5-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp510withNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_5_10, true);
  }

  /**
   * test whether clustering the Exp5-10 data set with using the clustering
   * number works
   */
  @Test(timeout = 3600000)
  public void testExp510withoutNumber() {
    this.dataClusterExample(ClusteringExampleDatasets.EXP_5_10, false);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final IntMatrix1D _randomIntMatrix(final int m,
      final int n, final Random random) {
    final int[] data;
    int i;

    i = m * n;
    data = new int[i];
    for (; (--i) >= 0;) {
      data[i] = random.nextInt();
    }

    return new IntMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final LongMatrix1D _randomLongMatrix(final int m,
      final int n, final Random random) {
    final long[] data;
    int i;

    i = m * n;
    data = new long[i];
    for (; (--i) >= 0;) {
      data[i] = random.nextLong();
    }

    return new LongMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final DoubleMatrix1D _randomDoubleMatrix(final int m,
      final int n, final Random random) {
    final double[] data;
    double dvalue;
    int i;

    i = m * n;
    data = new double[i];
    for (; (--i) >= 0;) {
      do {
        dvalue = Double.longBitsToDouble(random.nextLong());
      } while ((dvalue != dvalue) || (dvalue <= Double.NEGATIVE_INFINITY)
          || (dvalue >= Double.POSITIVE_INFINITY));
      data[i] = dvalue;
    }

    return new DoubleMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  static final AbstractMatrix _randomMatrix(final int m, final int n,
      final Random random) {
    switch (random.nextInt(3)) {
      case 0: {
        return ExampleDataClustererTest._randomIntMatrix(m, n, random);
      }
      case 1: {
        return ExampleDataClustererTest._randomLongMatrix(m, n, random);
      }
      default: {
        return ExampleDataClustererTest._randomDoubleMatrix(m, n, random);
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  protected final void setRandomData(final IClusteringJobBuilder builder,
      final int elementCount, final Random random) {
    ((IDataClusteringJobBuilder) builder).setData(ExampleDataClustererTest
        ._randomMatrix(elementCount, (1 + random.nextInt(20)), random));
  }
}
