package test.junit.org.optimizationBenchmarking.utils.ml.classification;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerReducedErrorPruned;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka J48 Trainer */
public class WekaJ48TrainerReducedErrorPrunedTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaJ48TrainerReducedErrorPrunedTest() {
    super(WekaJ48TrainerReducedErrorPruned.getInstance());
  }

}
