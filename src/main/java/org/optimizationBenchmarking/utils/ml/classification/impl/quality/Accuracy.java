package org.optimizationBenchmarking.utils.ml.classification.impl.quality;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

/**
 * The reverse multi-class accuracy measure: {@code 0} means that all
 * samples have been classified correctly, {@code 1} means all have been
 * classified wrongly. This is actually {@code 1-accuracy}
 * (https://en.wikipedia.org/wiki/Accuracy_and_precision), but since we
 * want to always <em>minimize</em> quality measures, we do it like that.
 */
public final class Accuracy extends ClassifierQualityMeasure<Void> {

  /** create */
  private Accuracy() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final IClassifier classifier,
      final Void token, final ClassifiedSample[] trainingSamples) {
    int correct;

    correct = trainingSamples.length;
    for (final ClassifiedSample sample : trainingSamples) {
      if (sample.sampleClass == classifier
          .classify(sample.featureValues)) {
        --correct;
      }
    }

    return (((double) correct) / trainingSamples.length);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "(In)Accuracy"; //$NON-NLS-1$
  }
}
