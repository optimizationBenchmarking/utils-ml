package test.junit.org.optimizationBenchmarking.utils.ml.classification.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaREPTreeTrainerUnpruned;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a Weka REPTree Trainer */
public class WekaREPTreeTrainerUnprunedTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public WekaREPTreeTrainerUnprunedTest() {
    super(WekaREPTreeTrainerUnpruned.getInstance());
  }
}
