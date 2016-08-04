package test.junit.org.optimizationBenchmarking.utils.ml.classification;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerReducedErrorPrunedBinary;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerReducedErrorPrunedBinaryTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerReducedErrorPrunedBinaryTest() {
    super(WekaJ48TrainerReducedErrorPrunedBinary.getInstance());
  }

}
