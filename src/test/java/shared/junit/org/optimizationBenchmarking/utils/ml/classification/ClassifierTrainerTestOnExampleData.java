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

}
