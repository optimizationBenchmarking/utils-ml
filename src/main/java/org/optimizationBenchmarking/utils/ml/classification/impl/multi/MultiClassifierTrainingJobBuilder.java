package org.optimizationBenchmarking.utils.ml.classification.impl.multi;

import java.util.Collection;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;

/** the multi training job builder */
public final class MultiClassifierTrainingJobBuilder
    extends ClassifierTrainingJobBuilder {

  /** the trainers to use */
  Collection<IClassifierTrainer> m_trainers;

  /**
   * Create the job builder
   *
   * @param trainer
   *          the trainer
   */
  MultiClassifierTrainingJobBuilder(final ClassifierTrainer trainer) {
    super(trainer);
  }

  /**
   * check the trainer set
   *
   * @param trainers
   *          the trainer set
   */
  static final void _checkTrainers(
      final Collection<IClassifierTrainer> trainers) {
    if (trainers == null) {
      throw new IllegalArgumentException("Trainer set cannot be null."); //$NON-NLS-1$
    }
    if (trainers.isEmpty()) {
      throw new IllegalArgumentException("Trainer set cannot be empty.");//$NON-NLS-1$
    }
  }

  /**
   * Set the trainers to be used
   *
   * @param trainers
   *          the trainers
   * @return this builder
   */
  public final MultiClassifierTrainingJobBuilder setTrainers(
      final IClassifierTrainer... trainers) {
    if (trainers == null) {
      throw new IllegalArgumentException("Trainer set cannot be null."); //$NON-NLS-1$
    }
    if (trainers.length <= 0) {
      throw new IllegalArgumentException("Trainer set cannot be empty."); //$NON-NLS-1$
    }
    this.m_trainers = new ArrayListView<>(trainers, false);
    return this;
  }

  /**
   * Set the trainers to be used
   *
   * @param trainers
   *          the trainers
   * @return this builder
   */
  public final MultiClassifierTrainingJobBuilder setTrainers(
      final Collection<IClassifierTrainer> trainers) {
    MultiClassifierTrainingJobBuilder._checkTrainers(trainers);
    this.m_trainers = trainers;
    return this;
  }
}
