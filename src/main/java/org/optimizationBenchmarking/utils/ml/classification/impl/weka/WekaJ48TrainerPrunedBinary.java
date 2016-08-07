package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * A classifier trainer for error-reducing pruned J48 trees with binary
 * splits in Weka.
 */
public final class WekaJ48TrainerPrunedBinary extends _WekaJ48Trainer {

  /** The fitting method name */
  static final String METHOD = _WekaJ48Trainer.BASE_METHOD
      + " (Pruned, Binary)"; //$NON-NLS-1$

  /** create */
  WekaJ48TrainerPrunedBinary() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaJ48ClassifierTrainingJob(builder,
        _WekaJ48ClassifierTrainingJob.PRUNING_ON, true);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return WekaJ48TrainerPrunedBinary.METHOD;
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    return _WekaJ48Classifier._printDescription(textOut, textCase,
        _WekaJ48ClassifierTrainingJob.PRUNING_ON, true, true);
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaJ48TrainerPrunedBinary getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaJ48TrainerPrunedBinary INSTANCE = new WekaJ48TrainerPrunedBinary();
  }
}
