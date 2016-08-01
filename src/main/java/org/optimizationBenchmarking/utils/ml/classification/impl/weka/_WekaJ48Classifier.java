package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.WekaClassifierTreeAccessor;
import weka.core.Instance;

/**
 * The wrapper for Weka's {@link weka.classifiers.trees.J48} classifiers
 * classifier.
 */
final class _WekaJ48Classifier extends _WekaClassifier {

  /**
   * Create the Weka J48 classifier wrapper
   *
   * @param classifier
   *          the classifier
   * @param vector
   *          the attribute vector
   * @param instance
   *          to use
   */
  _WekaJ48Classifier(final Classifier classifier, final double[] vector,
      final Instance instance) {
    super(classifier, vector, instance);
  }

  /** {@inheritDoc} */
  @Override
  public final void render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaClassifierTreeAccessor.renderJ48Classifier(
        ((J48) (this.m_classifier)), renderer, textOutput);
  }
}
