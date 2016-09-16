package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.comparison.Compare;
import org.optimizationBenchmarking.utils.hash.HashUtils;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

/** A classification result record */
public final class ClassifierTrainingResult implements
    IClassifierTrainingResult, Comparable<IClassifierTrainingResult> {

  /** the classifier */
  public final IClassifier classifier;
  /** the quality of the classifier */
  public final double quality;
  /** the complexity of the classifier */
  public final double complexity;

  /**
   * Create the classifier training result record
   *
   * @param _classifier
   *          the classifier
   * @param _quality
   *          the quality
   * @param _complexity
   *          the complexity
   */
  public ClassifierTrainingResult(final IClassifier _classifier,
      final double _quality, final double _complexity) {
    super();

    if (_classifier == null) {
      throw new IllegalArgumentException("Classifier cannot be null."); //$NON-NLS-1$
    }
    if ((_quality < 0d) || (_quality != _quality)) {
      throw new IllegalArgumentException(
          "Invalid classifier quality: " + _quality);//$NON-NLS-1$
    }
    if ((_complexity < 0d) || (_complexity != _complexity)) {
      throw new IllegalArgumentException(
          "Invalid classifier complexity: " + _complexity);//$NON-NLS-1$
    }
    this.classifier = _classifier;
    this.quality = _quality;
    this.complexity = _complexity;
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

  /** {@inheritDoc} */
  @Override
  public final int hashCode() {
    return HashUtils.combineHashes(HashUtils.hashCode(this.classifier),
        HashUtils.combineHashes(HashUtils.hashCode(this.quality),
            HashUtils.hashCode(this.complexity)));

  }

  /** {@inheritDoc} */
  @Override
  public final boolean equals(final Object other) {
    final IClassifierTrainingResult otherRes;
    if (other == this) {
      return true;
    }
    if (other instanceof IClassifierTrainingResult) {
      otherRes = ((IClassifierTrainingResult) other);
      return ((Compare.equals(this.quality, otherRes.getQuality())
          && (Compare.equals(this.complexity, otherRes.getComplexity()))
          && Compare.equals(this.classifier, otherRes.getClassifier())));
    }
    return true;
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final IClassifierTrainingResult o) {
    int res;

    if (o == this) {
      return 0;
    }
    res = Compare.compare(this.quality, o.getQuality());
    if (res != 0) {
      return res;
    }
    res = Compare.compare(this.complexity, o.getComplexity());
    if (res != 0) {
      return res;
    }

    return Compare.compare(this, o);
  }

  /** {@inheritDoc} */
  @Override
  public final double getComplexity() {
    return this.complexity;
  }
}
