package examples.org.optimizationBenchmarking.utils.ml.classifying;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;

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
  /** the nominal example 1 */
  public static final ClassifierExampleDataset NOMINAL_1 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 0), //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 2, 0), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 1, 1), //
          new ClassifiedSample(1, 2, 1),//
  });
  /** the nominal example 2 */
  public static final ClassifierExampleDataset NOMINAL_2 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 0), //
          new ClassifiedSample(0, 0, 1), //
          new ClassifiedSample(0, 0, 2), //
          new ClassifiedSample(1, 1, 0), //
          new ClassifiedSample(1, 1, 1), //
          new ClassifiedSample(1, 1, 2),//
  });

  /** the nominal example 3 */
  public static final ClassifierExampleDataset NOMINAL_3 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 1, 1), //
          new ClassifiedSample(0, 1, 2), //
          new ClassifiedSample(1, 0, 0), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 0, 2),//
  });

  /** the nominal example 4 */
  public static final ClassifierExampleDataset NOMINAL_4 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 0, 1),//
  });

  /** the nominal example 5 */
  public static final ClassifierExampleDataset NOMINAL_5 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 1, 0, 1), //
          new ClassifiedSample(0, 1, 0, 2), //
          new ClassifiedSample(0, 1, 0, 3), //
          new ClassifiedSample(1, 0, 1, 4), //
          new ClassifiedSample(1, 0, 1, 5), //
          new ClassifiedSample(1, 0, 1, 6),//
  });

  /** the nominal example 6 */
  public static final ClassifierExampleDataset NOMINAL_6 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 1, 6, 0),//
  });

  /** the nominal example 7 */
  public static final ClassifierExampleDataset NOMINAL_7 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });

  /** the mixed example 1.1 */
  public static final ClassifierExampleDataset MIXED_1_1 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NUMERICAL, EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 0), //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 2, 0), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 1, 1), //
          new ClassifiedSample(1, 2, 1),//
  });

  /** the mixed example 1.2 */
  public static final ClassifierExampleDataset MIXED_1_2 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NUMERICAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 0), //
          new ClassifiedSample(0, 1, 0), //
          new ClassifiedSample(0, 2, 0), //
          new ClassifiedSample(1, 0, 1), //
          new ClassifiedSample(1, 1, 1), //
          new ClassifiedSample(1, 2, 1),//
  });

  /** the mixed example 7.1 */
  public static final ClassifierExampleDataset MIXED_7_1 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NUMERICAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });

  /** the mixed example 7.2 */
  public static final ClassifierExampleDataset MIXED_7_2 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NUMERICAL,
          EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });

  /** the mixed example 7.3 */
  public static final ClassifierExampleDataset MIXED_7_3 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NUMERICAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });

  /** the mixed example 7.4 */
  public static final ClassifierExampleDataset MIXED_7_4 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL, EFeatureType.NUMERICAL,
          EFeatureType.NOMINAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });

  /** the mixed example 7.5 */
  public static final ClassifierExampleDataset MIXED_7_5 = new ClassifierExampleDataset(//
      new EFeatureType[] { EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NOMINAL, EFeatureType.NOMINAL,
          EFeatureType.NUMERICAL }, //
      new ClassifiedSample[] { //
          new ClassifiedSample(0, 0, 1, 0, 1, 0), //
          new ClassifiedSample(0, 0, 1, 0, 2, 0), //
          new ClassifiedSample(0, 0, 1, 0, 3, 0), //
          new ClassifiedSample(1, 0, 0, 1, 4, 0), //
          new ClassifiedSample(1, 0, 0, 1, 5, 0), //
          new ClassifiedSample(1, 0, 0, 1, 6, 0),//
  });
}
