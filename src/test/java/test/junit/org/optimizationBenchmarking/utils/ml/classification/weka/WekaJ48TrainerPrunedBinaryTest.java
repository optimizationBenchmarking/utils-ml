package test.junit.org.optimizationBenchmarking.utils.ml.classification.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerPrunedBinary;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerPrunedBinaryTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerPrunedBinaryTest() {
    super(WekaJ48TrainerPrunedBinary.getInstance());
  }
}
