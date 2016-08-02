package org.optimizationBenchmarking.utils.ml.classification.impl.multi;

import java.util.Collection;

import org.optimizationBenchmarking.utils.ml.classification.impl.DefaultClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** A classifier trainer multiple classifiers. */
public final class MultiClassifierTrainer extends ClassifierTrainer {

  /** The multi-classifier method name */
  static final String METHOD = "Multi Classifier Trainer"; //$NON-NLS-1$

  /** create */
  MultiClassifierTrainer() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder) {
    final MultiClassifierTrainingJobBuilder multiBuilder;
    final Collection<IClassifierTrainer> defaults;

    multiBuilder = ((MultiClassifierTrainingJobBuilder) builder);
    if (multiBuilder.m_trainers == null) {
      defaults = DefaultClassifierTrainer.getAllInstance();
      if ((defaults != null) && (defaults.size() > 0)) {
        multiBuilder.setTrainers(defaults);
      }
    }
    return new _MultiClassifierTrainingJob(multiBuilder);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return MultiClassifierTrainer.METHOD;
  }

  /** {@inheritDoc} */
  @Override
  public final MultiClassifierTrainingJobBuilder use() {
    return new MultiClassifierTrainingJobBuilder(this);
  }

  /**
   * Obtain the globally shared instance of the multi-classifier trainer
   *
   * @return the globally shared instance of the multi-classifier trainer
   */
  public static final MultiClassifierTrainer getInstance() {
    return __MultiClassifierTrainerHolder.INSTANCE;
  }

  /** the instance holder */
  private static final class __MultiClassifierTrainerHolder {
    /** the globally shared instance */
    static final MultiClassifierTrainer INSTANCE = new MultiClassifierTrainer();
  }
}
