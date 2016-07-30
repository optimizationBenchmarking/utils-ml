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
    final int numClasses;
    numClasses = ClassificationTools.getClassCount(trainingSamples);
    return new int[numClasses][numClasses];
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
