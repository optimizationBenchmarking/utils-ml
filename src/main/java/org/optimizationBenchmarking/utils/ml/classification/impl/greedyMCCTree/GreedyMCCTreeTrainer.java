package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** the trainer for greedy mcc trees */
public class GreedyMCCTreeTrainer extends ClassifierTrainer {

  /** the greedy mcc tree name */
  static final String NAME = "Greedy MCC Tree"; //$NON-NLS-1$

  /** create */
  GreedyMCCTreeTrainer() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _GreedyMCCTreeTrainingJob(builder);
  }

  /** {@inheritDoc} */
  @Override
  public final boolean canUse() {
    return true;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return GreedyMCCTreeTrainer.NAME;
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final GreedyMCCTreeTrainer getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final GreedyMCCTreeTrainer INSTANCE = new GreedyMCCTreeTrainer();
  }
}
