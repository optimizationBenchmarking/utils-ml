package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

/** The confusion matrix token. */
public final class ConfusionMatrix extends ClassifiedSampleInfo {

  /** the confusion matrix */
  private final int[][] m_matrix;

  /**
   * create the confusion matrix token
   *
   * @param samples
   *          the samples
   */
  public ConfusionMatrix(final ClassifiedSample[] samples) {
    super(samples);

    final int size;
    size = this.getClassCount();
    this.m_matrix = new int[size][size];
  }

  /**
   * create a confusion matrix from the given matrix rows
   *
   * @param rows
   *          the rows
   */
  public ConfusionMatrix(final int[]... rows) {
    super(ConfusionMatrix.__rowsToSizes(rows));
    this.m_matrix = rows;
  }

  /**
   * transform the given confusion matrix into a sizes array
   *
   * @param rows
   *          the matrix rows
   * @return the sizes array
   */
  private static final int[] __rowsToSizes(final int[][] rows) {
    final int[] sizes;
    int index;

    sizes = new int[rows.length];
    index = (-1);
    for (final int[] row : rows) {
      ++index;
      for (final int cell : row) {
        sizes[index] += cell;
      }
    }
    return sizes;
  }

  /**
   * Fill in the confusion matrix
   *
   * @param classifier
   *          the classifier
   * @param trainingSamples
   *          the training samples
   */
  public final void fillInConfusionMatrix(final IClassifier classifier,
      final ClassifiedSample[] trainingSamples) {
    for (final int[] row : this.m_matrix) {
      Arrays.fill(row, 0);
    }
    for (final ClassifiedSample sample : trainingSamples) {
      ++this.m_matrix[this.getClassIndex(sample.sampleClass)][this
          .getClassIndex(classifier.classify(sample.featureValues))];
    }
  }

  /**
   * Get the confusion value for a given set of index classes
   *
   * @param isClass
   *          the actual index class
   * @param classifiedClass
   *          the index class returned by the classifier
   * @return the confusion value
   */
  public final int getConfusionForIndexClasses(final int isClass,
      final int classifiedClass) {
    return this.m_matrix[isClass][classifiedClass];
  }

  /**
   * Get the confusion value for a given set of sample classes
   *
   * @param isClass
   *          the actual sample class
   * @param classifiedClass
   *          the sample class returned by the classifier
   * @return the confusion value
   */
  public final int getConfusionForSampleClasses(final int isClass,
      final int classifiedClass) {
    return this.getConfusionForIndexClasses(this.getClassIndex(isClass),
        this.getClassIndex(classifiedClass));
  }
}
