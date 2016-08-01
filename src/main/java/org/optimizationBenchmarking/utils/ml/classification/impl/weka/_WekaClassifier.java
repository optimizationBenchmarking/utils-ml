package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

import weka.classifiers.Classifier;
import weka.core.Instance;

/** The base class for wrapping Weka classifiers */
abstract class _WekaClassifier implements IClassifier {

  /** the internal classifier */
  final Classifier m_classifier;

  /** the vector to use */
  private final double[] m_vector;

  /** the instance to use */
  private final Instance m_instance;

  /**
   * Create the weka classifier wrapper
   *
   * @param classifier
   *          the classifier
   * @param vector
   *          the attribute vector
   * @param instance
   *          to use
   */
  _WekaClassifier(final Classifier classifier, final double[] vector,
      final Instance instance) {
    super();
    if (classifier == null) {
      throw new IllegalArgumentException("Classifier must not be null."); //$NON-NLS-1$
    }
    if (vector == null) {
      throw new IllegalArgumentException("Raw vector must not be null."); //$NON-NLS-1$
    }
    if (instance == null) {
      throw new IllegalArgumentException("Instance must not be null."); //$NON-NLS-1$
    }

    this.m_classifier = classifier;
    this.m_vector = vector;
    this.m_instance = instance;
  }

  /** {@inheritDoc} */
  @Override
  public final int classify(final double[] features) {
    System.arraycopy(features, 0, this.m_vector, 0, features.length);
    try {
      return ((int) (0.5d
          + this.m_classifier.classifyInstance(this.m_instance)));
    } catch (final Exception exception) {
      throw new IllegalArgumentException(
          "Error when trying to classify instance " //$NON-NLS-1$
              + Arrays.toString(features),
          exception);
    }
  }
}
