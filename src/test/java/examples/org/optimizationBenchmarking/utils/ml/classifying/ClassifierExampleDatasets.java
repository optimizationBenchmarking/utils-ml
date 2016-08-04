package examples.org.optimizationBenchmarking.utils.ml.classifying;

import examples.org.optimizationBenchmarking.utils.ml.clustering.ClusteringExampleDatasets;
import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierExampleDataset;

/** The loader for the classifying example data sets. */
public final class ClassifierExampleDatasets {

  /** the simple example 2 */
  public static final ClassifierExampleDataset SIMPLE_2 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.SIMPLE_2);
  /** the simple example 3 */
  public static final ClassifierExampleDataset SIMPLE_3 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.SIMPLE_3);
  /** the simple example 4 */
  public static final ClassifierExampleDataset SIMPLE_4 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.SIMPLE_4);
  /** the exponential example 3/2 */
  public static final ClassifierExampleDataset EXP_3_2 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_3_2);
  /** the exponential example 4/2 */
  public static final ClassifierExampleDataset EXP_4_2 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_4_2);
  /** the exponential example 5/2 */
  public static final ClassifierExampleDataset EXP_5_2 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_5_2);
  /** the exponential example 6/2 */
  public static final ClassifierExampleDataset EXP_6_2 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_6_2);
  /** the exponential example 3/10 */
  public static final ClassifierExampleDataset EXP_3_10 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_3_10);
  /** the exponential example 4/10 */
  public static final ClassifierExampleDataset EXP_4_10 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_4_10);
  /** the exponential example 5/10 */
  public static final ClassifierExampleDataset EXP_5_10 = new ClassifierExampleDataset(
      ClusteringExampleDatasets.EXP_5_10);
  /** the iris example */
  public static final ClassifierExampleDataset IRIS = new ClassifierExampleDataset(
      ClusteringExampleDatasets.IRIS);
}
