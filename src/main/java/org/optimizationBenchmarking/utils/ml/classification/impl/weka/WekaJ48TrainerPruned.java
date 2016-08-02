package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** A classifier trainer for error-reducing pruned J48 trees in Weka. */
public final class WekaJ48TrainerPruned extends _WekaJ48Trainer {

  /** The fitting method name */
  static final String METHOD = _WekaJ48Trainer.BASE_METHOD + " (Pruned)"; //$NON-NLS-1$

  /** create */
  WekaJ48TrainerPruned() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaJ48ClassifierTrainingJob(builder,
        _WekaJ48ClassifierTrainingJob.PRUNING_REDUCED_ERROR, false);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return WekaJ48TrainerPruned.METHOD;
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaJ48TrainerPruned getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaJ48TrainerPruned INSTANCE = new WekaJ48TrainerPruned();
  }
}
