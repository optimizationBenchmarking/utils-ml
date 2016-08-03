package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/** a classifier training job wrapping around Weka */
class _WekaJ48ClassifierTrainingJob
    extends _WekaClassifierTrainingJob<J48> {

  /** pruning is turned off */
  static final int PRUNING_OFF = 0;
  /** pruning is turned on */
  static final int PRUNING_ON = (_WekaJ48ClassifierTrainingJob.PRUNING_OFF
      + 1);
  /** pruning is turned on and uses reduced error method */
  static final int PRUNING_REDUCED_ERROR = (_WekaJ48ClassifierTrainingJob.PRUNING_ON
      + 1);

  /** do we perform pruning? */
  private final int m_pruning;
  /** should we always split binary? */
  private final boolean m_binary;

  /**
   * Create the weka classifier training job
   *
   * @param builder
   *          the builder
   * @param pruning
   *          should we perform pruning
   * @param binary
   *          should we always split binary?
   */
  _WekaJ48ClassifierTrainingJob(final ClassifierTrainingJobBuilder builder,
      final int pruning, final boolean binary) {
    super(builder);
    this.m_pruning = pruning;
    this.m_binary = binary;
  }

  /** {@inheritDoc} */
  @Override
  final J48 _train(final Instances instances) {
    final J48 classifier;

    classifier = new J48();
    switch (this.m_pruning) {
      case PRUNING_ON: {
        classifier.setUnpruned(false);
        classifier.setSubtreeRaising(true);
        classifier.setReducedErrorPruning(false);
        break;
      }
      case PRUNING_REDUCED_ERROR: {
        classifier.setUnpruned(false);
        classifier.setSubtreeRaising(true);
        classifier.setReducedErrorPruning(true);
        break;
      }
      default: {
        classifier.setUnpruned(true);
        break;
      }
    }
    if (this.m_binary) {
      classifier.setBinarySplits(true);
    }

    try {
      classifier.buildClassifier(instances);
    } catch (final Throwable error) {
      throw new IllegalStateException((//
          "Error while trying to train a J48 classifier " + //$NON-NLS-1$
              (this.m_binary ? "with" : "without") + //$NON-NLS-1$//$NON-NLS-2$
              " binary splits and " + //$NON-NLS-1$
              (((this.m_pruning == _WekaJ48ClassifierTrainingJob.PRUNING_ON)
                  ? "with" //$NON-NLS-1$
                  : ((this.m_pruning == _WekaJ48ClassifierTrainingJob.PRUNING_OFF)
                      ? "without" //$NON-NLS-1$
                      : "with error-reduced")))//$NON-NLS-1$
              + " pruning to " + //$NON-NLS-1$
              +instances.size() + " data samples."), //$NON-NLS-1$
          error);
    }

    return classifier;
  }

  /** {@inheritDoc} */
  @Override
  final _WekaClassifier<J48> _createClassifier(final J48 classifier,
      final double[] vector, final Instance instance) {
    return new _WekaJ48Classifier(classifier, vector, instance);
  }

  /** {@inheritDoc} */
  @Override
  protected final String getJobName() {
    switch (this.m_pruning) {
      case PRUNING_ON: {
        return (this.m_binary ? WekaJ48TrainerPrunedBinary.METHOD
            : WekaJ48TrainerPruned.METHOD);
      }
      case PRUNING_REDUCED_ERROR: {
        return (this.m_binary
            ? WekaJ48TrainerReducedErrorPrunedBinary.METHOD
            : WekaJ48TrainerReducedErrorPruned.METHOD);
      }
      default: {
        return (this.m_binary ? WekaJ48TrainerUnprunedBinary.METHOD
            : WekaJ48TrainerUnpruned.METHOD);
      }
    }
  }
}
