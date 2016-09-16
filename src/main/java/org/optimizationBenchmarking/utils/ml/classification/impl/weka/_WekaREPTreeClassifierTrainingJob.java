package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;

import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.WekaTreeAccessor;
import weka.core.Instances;

/** a classifier training job wrapping around Weka for REPTree */
class _WekaREPTreeClassifierTrainingJob
    extends _WekaClassifierTrainingJob<REPTree> {

  /** do we perform pruning? */
  private final boolean m_pruning;

  /**
   * Create the weka classifier training job
   *
   * @param builder
   *          the builder
   * @param pruning
   *          should we perform pruning
   */
  _WekaREPTreeClassifierTrainingJob(
      final ClassifierTrainingJobBuilder builder, final boolean pruning) {
    super(builder);
    this.m_pruning = pruning;
  }

  /** {@inheritDoc} */
  @Override
  final REPTree _train(final Instances instances) {
    final REPTree classifier;

    classifier = new REPTree();
    classifier.setNoPruning(!(this.m_pruning));

    try {
      classifier.buildClassifier(instances);
    } catch (final Throwable error) {
      throw new IllegalStateException((//
          "Error while trying to train a REPTree classifier " + //$NON-NLS-1$
              (this.m_pruning ? "with" : "without") + //$NON-NLS-1$//$NON-NLS-2$
              " pruning to " + //$NON-NLS-1$
              +instances.size() + " data samples."), //$NON-NLS-1$
          error);
    }

    return classifier;
  }

  /** {@inheritDoc} */
  @Override
  final _WekaClassifier<REPTree> _createClassifier(
      final REPTree classifier, final _InternalInstance instance) {
    return new _WekaREPTreeClassifier(this.m_selectedFeatures, classifier,
        instance);
  }

  /** {@inheritDoc} */
  @Override
  protected final String getJobName() {
    return (this.m_pruning ? WekaREPTreeTrainerPruned.METHOD
        : WekaREPTreeTrainerUnpruned.METHOD);
  }

  /** {@inheritDoc} */
  @Override
  final double _getComplexity(final REPTree classifier) {
    return WekaTreeAccessor.getREPTreeComplexity(classifier);
  }
}
