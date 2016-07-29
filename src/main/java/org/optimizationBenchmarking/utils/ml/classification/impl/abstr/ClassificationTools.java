package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** Some simple tools for classification. */
public final class ClassificationTools {

  /**
   * Can we cluster the given samples trivially? If so, return a job which
   * just directly returns a fixed result. Otherwise return {@code null}.
   *
   * @param samples
   *          the samples
   * @return the job if classification can be done trivially, or
   *         {@code null} if trivial classifying is not possible
   */
  public static final IClassifierTrainingJob canClusterTrivially(
      final ClassifiedSample[] samples) {
    int minClass, maxClass;

    // check whether there is more than one class
    maxClass = (-1);
    minClass = Integer.MAX_VALUE;
    for (final ClassifiedSample sample : samples) {
      if (sample.sampleClass < minClass) {
        minClass = sample.sampleClass;
      }
      if (sample.sampleClass > maxClass) {
        maxClass = sample.sampleClass;
      }
    }

    if (minClass >= maxClass) {
      return new _AllTheSameClass(minClass);
    }

    return null;
  }

  /** do not call */
  private ClassificationTools() {
    ErrorUtils.doNotCall();
  }
}
