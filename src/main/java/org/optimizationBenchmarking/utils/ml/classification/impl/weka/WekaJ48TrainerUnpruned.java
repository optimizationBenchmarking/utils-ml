package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** A classifier trainer for unpruned J48 trees in Weka. */
public final class WekaJ48TrainerUnpruned extends _WekaJ48Trainer {

  /** create */
  WekaJ48TrainerUnpruned() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaJ48ClassifierTrainingJob(builder,
        _WekaJ48ClassifierTrainingJob.PRUNING_OFF, false);
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaJ48TrainerUnpruned getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaJ48TrainerUnpruned INSTANCE = new WekaJ48TrainerUnpruned();
  }
}