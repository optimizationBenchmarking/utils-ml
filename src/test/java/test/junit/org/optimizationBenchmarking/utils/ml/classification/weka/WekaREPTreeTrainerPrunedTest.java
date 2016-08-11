package test.junit.org.optimizationBenchmarking.utils.ml.classification.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaREPTreeTrainerPruned;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka REPTree Trainer */
public class WekaREPTreeTrainerPrunedTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaREPTreeTrainerPrunedTest() {
    super(WekaREPTreeTrainerPruned.getInstance());
  }
}
