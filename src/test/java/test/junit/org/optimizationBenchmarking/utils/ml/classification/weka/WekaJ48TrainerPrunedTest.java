package test.junit.org.optimizationBenchmarking.utils.ml.classification.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerPruned;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerPrunedTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerPrunedTest() {
    super(WekaJ48TrainerPruned.getInstance());
  }
}
