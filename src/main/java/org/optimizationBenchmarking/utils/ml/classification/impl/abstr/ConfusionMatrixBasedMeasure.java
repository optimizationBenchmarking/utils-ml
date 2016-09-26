package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

/** A confusion matrix-based measure */
public class ConfusionMatrixBasedMeasure
    extends ClassifierQualityMeasure<ConfusionMatrix> {

  /** create */
  protected ConfusionMatrixBasedMeasure() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public double evaluate(final IClassifier classifier,
      final ConfusionMatrix token,
      final ClassifiedSample[] trainingSamples) {
    return 0.5d;
  }

  /** {@inheritDoc} */
  @Override
  public final ConfusionMatrix createToken(
      final ClassifiedSample[] trainingSamples) {
    return new ConfusionMatrix(trainingSamples);
  }
}
