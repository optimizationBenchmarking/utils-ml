package shared.junit.org.optimizationBenchmarking.utils.ml.clustering;

import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DistanceMatrixBuilderJob;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.IAggregate;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.EuclideanDistance;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.MeasureBasedDistanceMatrixBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringJobBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusteringJobBuilder;

import examples.org.optimizationBenchmarking.utils.ml.clustering.ClusteringExampleDatasets;

/** The test for data clustering. */
@Ignore
public abstract class ExampleDistanceClustererTest
    extends DistanceClustererTest {

  /**
   * create
   *
   * @param clusterer
   *          the clusterer
   */
  public ExampleDistanceClustererTest(final IDistanceClusterer clusterer) {
    super(clusterer);
  }

  /** {@inheritDoc} */
  @Override
  protected final IMatrix dataMatrixToDistanceMatrix(
      final IMatrix dataMatrix) {
    return new MeasureBasedDistanceMatrixBuilder(dataMatrix,
        new EuclideanDistance()).call();
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

  /** {@inheritDoc} */
  @Override
  protected final void setRandomData(final IClusteringJobBuilder builder,
      final int elementCount, final Random random) {
    IMatrix distance;
    double dvalue;
    int i, j;

    looper: for (;;) {
      if (random.nextBoolean()) {
        distance = new __RandomDistanceMatrixBuilder(elementCount, random)
            .call();
      } else {
        distance = new MeasureBasedDistanceMatrixBuilder(//
            ExampleDataClustererTest._randomMatrix(elementCount,
                (1 + random.nextInt(20)), random),
            new EuclideanDistance()).call();
      }

      for (i = elementCount; (--i) >= 0;) {
        for (j = elementCount; (--j) >= 0;) {
          dvalue = distance.getDouble(i, j);
          if ((dvalue != dvalue) || (dvalue <= Double.NEGATIVE_INFINITY)
              || (dvalue >= Double.POSITIVE_INFINITY)) {
            continue looper;
          }
        }
      }
      break looper;
    }

    ((IDistanceClusteringJobBuilder) builder).setDistanceMatrix(distance);
  }

  /** the random distance matrix builder */
  private static final class __RandomDistanceMatrixBuilder
      extends DistanceMatrixBuilderJob {
    /** the m */
    final int m_m;
    /** the random number generator */
    final Random m_random;
    /** the mode */
    final int m_mode;

    /**
     * create
     *
     * @param m
     *          the m
     * @param random
     *          the random
     */
    __RandomDistanceMatrixBuilder(final int m, final Random random) {
      super();
      this.m_m = m;
      this.m_mode = random.nextInt(3);
      this.m_random = random;
    }

    /** {@inheritDoc} */
    @Override
    protected final int getElementCount() {
      return this.m_m;
    }

    /** {@inheritDoc} */
    @Override
    protected final void setDistance(final int i, final int j,
        final IAggregate appendTo) {
      double dvalue;
      switch (this.m_mode) {
        case 0: {
          appendTo.append(this.m_random.nextInt());
          return;
        }
        case 1: {
          appendTo.append(this.m_random.nextLong());
          return;
        }
        default: {
          do {
            dvalue = Double.longBitsToDouble(this.m_random.nextLong());
          } while ((dvalue != dvalue)
              || (dvalue <= Double.NEGATIVE_INFINITY)
              || (dvalue >= Double.POSITIVE_INFINITY));
          appendTo.append(dvalue);
        }
      }
    }
  }
}
