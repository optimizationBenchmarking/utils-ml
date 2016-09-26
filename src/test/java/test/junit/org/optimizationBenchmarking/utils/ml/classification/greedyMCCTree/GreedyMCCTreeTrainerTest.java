package test.junit.org.optimizationBenchmarking.utils.ml.classification.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree.GreedyMCCTreeTrainer;

import shared.junit.org.optimizationBenchmarking.utils.ml.classification.ClassifierTrainerTestOnExampleData;

/** The tests for a greedy mcc tree Trainer */
public class GreedyMCCTreeTrainerTest
    extends ClassifierTrainerTestOnExampleData {

  /** create */
  public GreedyMCCTreeTrainerTest() {
    super(GreedyMCCTreeTrainer.getInstance());
  }

}
