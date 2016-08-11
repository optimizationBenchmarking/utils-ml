package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * A classifier trainer for error-reducing pruned REPTree trees in Weka.
 */
public final class WekaREPTreeTrainerPruned extends _WekaREPTreeTrainer {

  /** The fitting method name */
  static final String METHOD = _WekaREPTreeTrainer.BASE_METHOD
      + " (Pruned)"; //$NON-NLS-1$

  /** create */
  WekaREPTreeTrainerPruned() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    return new _WekaREPTreeClassifierTrainingJob(builder, true);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    return _WekaREPTreeClassifier._printDescription(textOut, textCase,
        true, true);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return WekaREPTreeTrainerPruned.METHOD;
  }

  /**
   * Get the globally shared instance of this trainer
   *
   * @return the globally shared instance of this trainer
   */
  public static final WekaREPTreeTrainerPruned getInstance() {
    return __InstanceHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __InstanceHolder {
    /** the shared instance */
    static final WekaREPTreeTrainerPruned INSTANCE = new WekaREPTreeTrainerPruned();
  }
}
