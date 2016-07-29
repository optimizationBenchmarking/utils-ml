package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

/** A confusion matrix-based measure */
public class ConfusionMatrixBasedMeasure
    extends ClassifierQualityMeasure<int[][]> {

  /** create */
  protected ConfusionMatrixBasedMeasure() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public double evaluate(final IClassifier classifier, final int[][] token,
      final ClassifiedSample[] trainingSamples) {
    return 0.5d;
  }

  /** {@inheritDoc} */
  @Override
  public final int[][] createToken(
      final ClassifiedSample[] trainingSamples) {
    int maxClass;

    maxClass = (-1);
    for (final ClassifiedSample sample : trainingSamples) {
      if (sample.sampleClass > maxClass) {
        maxClass = sample.sampleClass;
      }
    }

    ++maxClass;
    return new int[maxClass][maxClass];
  }

  /**
   * Fill in the confusion matrix
   *
   * @param classifier
   *          the classifier
   * @param trainingSamples
   *          the training samples
   * @param token
   *          the matrix to fill in
   */
  protected static final void fillInConfusionMatrix(
      final IClassifier classifier,
      final ClassifiedSample[] trainingSamples, final int[][] token) {
    for (final int[] row : token) {
      Arrays.fill(row, 0);
    }
    for (final ClassifiedSample sample : trainingSamples) {
      ++token[sample.sampleClass][classifier
          .classify(sample.featureValues)];
    }
  }
}
