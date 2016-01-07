package shared.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.junit.Ignore;
import org.junit.Test;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.EuclideanDistance;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.MeasureBasedDistanceMatrixBuilder;

import examples.org.optimizationBenchmarking.utils.ml.clustering.ClusteringExampleDatasets;

/** The test for data clustering. */
@Ignore
public abstract class ExampleDistanceClustererTest
    extends DistanceClustererTest {

  /** create */
  public ExampleDistanceClustererTest() {
    super();
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
}
