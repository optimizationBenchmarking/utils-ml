package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** A classifier trainer for basically pruned J48 trees in Weka. */
public final class WekaJ48TrainerReducedErrorPruned
    extends _WekaJ48Trainer {

  /** The fitting method name */
  static final String METHOD = _WekaJ48Trainer.BASE_METHOD
      + " (Reduced-Error Pruned)"; //$NON-NLS-1$

  /** create */
  WekaJ48TrainerReducedErrorPruned() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaJ48ClassifierTrainingJob(builder,
        _WekaJ48ClassifierTrainingJob.PRUNING_ON, false);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return WekaJ48TrainerReducedErrorPruned.METHOD;
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaJ48TrainerReducedErrorPruned getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaJ48TrainerReducedErrorPruned INSTANCE = new WekaJ48TrainerReducedErrorPruned();
  }
}
