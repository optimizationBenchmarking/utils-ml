package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

/** A classification result record */
public final class ClassifierTrainingResult
    implements IClassifierTrainingResult {

  /** the classifier */
  public final IClassifier classifier;
  /** the quality of the classifier */
  public final double quality;

  /**
   * Create the classifier training result record
   *
   * @param _classifier
   *          the classifier
   * @param _quality
   *          the quality
   */
  public ClassifierTrainingResult(final IClassifier _classifier,
      final double _quality) {
    super();

    if (_classifier == null) {
      throw new IllegalArgumentException("Classifier cannot be null."); //$NON-NLS-1$
    }
    if ((_quality < 0d) || (_quality != _quality)) {
      throw new IllegalArgumentException(
          "Invalid classifier quality: " + _quality);//$NON-NLS-1$
    }
    this.classifier = _classifier;
    this.quality = _quality;
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifier getClassifier() {
    return this.classifier;
  }

  /** {@inheritDoc} */
  @Override
  public final double getQuality() {
    return this.quality;
  }
}
