package test.junit.org.optimizationBenchmarking.utils.ml.classification;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerUnprunedBinary;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerUnprunedBinaryTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerUnprunedBinaryTest() {
    super(WekaJ48TrainerUnprunedBinary.getInstance());
  }

}
