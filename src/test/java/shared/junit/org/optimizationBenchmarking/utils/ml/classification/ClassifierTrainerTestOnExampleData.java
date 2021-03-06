package shared.junit.org.optimizationBenchmarking.utils.ml.classification;

import org.junit.Ignore;
import org.junit.Test;
import org.optimizationBenchmarking.utils.ml.classification.impl.quality.Accuracy;
import org.optimizationBenchmarking.utils.ml.classification.impl.quality.MCC;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;

import examples.org.optimizationBenchmarking.utils.ml.classifying.ClassifierExampleDatasets;

/**
 * A base class for tests for classifier trainers on example data
 */
@Ignore
public abstract class ClassifierTrainerTestOnExampleData
    extends ClassifierTrainerTest {

  /**
   * create the test
   *
   * @param classifier
   *          the classifier
   */
  protected ClassifierTrainerTestOnExampleData(
      final IClassifierTrainer classifier) {
    super(classifier);
  }

  /**
   * Apply a classifier trainer to the given sample data using the MCC
   * measure and verify the result
   *
   * @param samples
   *          the samples
   */
  protected void applyClassifierTrainerMCC(
      final ClassifierExampleDataset samples) {
    this.applyClassifierTrainer(samples, MCC.INSTANCE);
  }

  /**
   * Apply a classifier trainer to the given sample data using the Accuracy
   * measure and verify the result
   *
   * @param samples
   *          the samples
   */
  protected void applyClassifierTrainerAccuracy(
      final ClassifierExampleDataset samples) {
    this.applyClassifierTrainer(samples, Accuracy.INSTANCE);
  }

  /** test on the IRIS data by using MCC */
  @Test(timeout = 3600000)
  public void test_IRIS_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.IRIS);
  }

  /** test on the IRIS data by using MCC */
  @Test(timeout = 3600000)
  public void test_IRIS_Accuracy() {
    this.applyClassifierTrainerAccuracy(ClassifierExampleDatasets.IRIS);
  }

  /** test on the SIMPLE_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.SIMPLE_2);
  }

  /** test on the SIMPLE_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.SIMPLE_2);
  }

  /** test on the SIMPLE_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_3_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.SIMPLE_3);
  }

  /** test on the SIMPLE_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_3_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.SIMPLE_3);
  }

  /** test on the SIMPLE_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_4_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.SIMPLE_4);
  }

  /** test on the SIMPLE_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_SIMPLE_4_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.SIMPLE_4);
  }

  /** test on the EXP_3_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_3_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_3_2);
  }

  /** test on the EXP_3_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_3_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(ClassifierExampleDatasets.EXP_3_2);
  }

  /** test on the EXP_4_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_4_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_4_2);
  }

  /** test on the EXP_4_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_4_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(ClassifierExampleDatasets.EXP_4_2);
  }

  /** test on the EXP_5_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_5_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_5_2);
  }

  /** test on the EXP_5_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_5_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(ClassifierExampleDatasets.EXP_5_2);
  }

  /** test on the EXP_6_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_6_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_6_2);
  }

  /** test on the EXP_6_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_6_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(ClassifierExampleDatasets.EXP_6_2);
  }

  /** test on the EXP_3_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_3_10_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_3_10);
  }

  /** test on the EXP_3_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_3_10_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.EXP_3_10);
  }

  /** test on the EXP_4_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_4_10_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_4_10);
  }

  /** test on the EXP_4_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_4_10_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.EXP_4_10);
  }

  /** test on the EXP_5_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_5_10_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.EXP_5_10);
  }

  /** test on the EXP_5_10 data by using MCC */
  @Test(timeout = 3600000)
  public void test_EXP_5_10_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.EXP_5_10);
  }

  /** test on the NOMINAL_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_1_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_1);
  }

  /** test on the NOMINAL_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_1_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_1);
  }

  /** test on the NOMINAL_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_2);
  }

  /** test on the NOMINAL_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_2);
  }

  /** test on the NOMINAL_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_3_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_3);
  }

  /** test on the NOMINAL_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_3_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_3);
  }

  /** test on the NOMINAL_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_4_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_4);
  }

  /** test on the NOMINAL_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_4_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_4);
  }

  /** test on the NOMINAL_5 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_5_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_5);
  }

  /** test on the NOMINAL_5 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_5_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_5);
  }

  /** test on the NOMINAL_6 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_6_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_6);
  }

  /** test on the NOMINAL_6 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_6_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_6);
  }

  /** test on the NOMINAL_7 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_7_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.NOMINAL_7);
  }

  /** test on the NOMINAL_7 data by using MCC */
  @Test(timeout = 3600000)
  public void test_NOMINAL_7_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.NOMINAL_7);
  }

  /** test on the MIXED_1_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_1_1_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_1_1);
  }

  /** test on the MIXED_1_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_1_1_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_1_1);
  }

  /** test on the MIXED_1_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_1_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_1_2);
  }

  /** test on the MIXED_1_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_1_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_1_2);
  }

  /** test on the MIXED_7_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_1_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_7_1);
  }

  /** test on the MIXED_7_1 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_1_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_7_1);
  }

  /** test on the MIXED_7_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_2_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_7_2);
  }

  /** test on the MIXED_7_2 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_2_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_7_2);
  }

  /** test on the MIXED_7_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_3_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_7_3);
  }

  /** test on the MIXED_7_3 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_3_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_7_3);
  }

  /** test on the MIXED_7_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_4_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_7_4);
  }

  /** test on the MIXED_7_4 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_4_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_7_4);
  }

  /** test on the MIXED_7_5 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_5_MCC() {
    this.applyClassifierTrainerMCC(ClassifierExampleDatasets.MIXED_7_5);
  }

  /** test on the MIXED_7_5 data by using MCC */
  @Test(timeout = 3600000)
  public void test_MIXED_7_5_Accuracy() {
    this.applyClassifierTrainerAccuracy(
        ClassifierExampleDatasets.MIXED_7_5);
  }
}
