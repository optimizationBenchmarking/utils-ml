package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/**
 * A classifier trainer for basically pruned J48 trees with binary splits
 * in Weka.
 */
public final class WekaJ48TrainerReducedErrorPrunedBinary
    extends _WekaJ48Trainer {

  /** The fitting method name */
  static final String METHOD = _WekaJ48Trainer.BASE_METHOD
      + " (Reduced-Error Pruned, Binary)"; //$NON-NLS-1$

  /** create */
  WekaJ48TrainerReducedErrorPrunedBinary() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaJ48ClassifierTrainingJob(builder,
        _WekaJ48ClassifierTrainingJob.PRUNING_REDUCED_ERROR, true);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return WekaJ48TrainerReducedErrorPrunedBinary.METHOD;
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaJ48TrainerReducedErrorPrunedBinary getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaJ48TrainerReducedErrorPrunedBinary INSTANCE = new WekaJ48TrainerReducedErrorPrunedBinary();
  }
}
