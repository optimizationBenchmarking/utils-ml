package test.junit.org.optimizationBenchmarking.utils.ml.classification.weka;

import org.junit.Ignore;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerUnpruned;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerUnprunedTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerUnprunedTest() {
    super(WekaJ48TrainerUnpruned.getInstance());
  }

  /** {@inheritDoc} */
  @Override
  @Ignore
  public void test_MIXED_1_1_MCC() {
    // Weka's J48 cannot deal with this appropriately
  }

  /** {@inheritDoc} */
  @Override
  @Ignore
  public void test_MIXED_1_1_Accuracy() {
    // Weka's J48 cannot deal with this appropriately
  }
}
