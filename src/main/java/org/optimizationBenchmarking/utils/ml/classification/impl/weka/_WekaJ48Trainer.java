package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainer;
import org.optimizationBenchmarking.utils.reflection.ReflectionUtils;

/** A classifier trainer for Weka's J48 trainers. */
abstract class _WekaJ48Trainer extends ClassifierTrainer {

  /** the error */
  private final Throwable m_error;

  /** create */
  _WekaJ48Trainer() {
    super();

    Throwable cannot;

    cannot = null;
    try {
      ReflectionUtils.ensureClassesAreLoaded(//
          "weka.classifiers.Classifier", //$NON-NLS-1$
          "weka.classifiers.trees.J48", //$NON-NLS-1$
          "weka.core.Instance", //$NON-NLS-1$
          "weka.core.Instances", //$NON-NLS-1$
          "weka.core.Attribute", //$NON-NLS-1$
          "weka.core.DenseInstance" //$NON-NLS-1$
      );

    } catch (final Throwable error) {
      cannot = error;
    }

    this.m_error = cannot;
  }

  /** {@inheritDoc} */
  @Override
  public final boolean canUse() {
    return (this.m_error == null);
  }

  /** {@inheritDoc} */
  @Override
  public final void checkCanUse() {
    if (this.m_error != null) {
      throw new UnsupportedOperationException(//
          this.toString()
              + " cannot be used since required classes are missing in the classpath.", //$NON-NLS-1$
          this.m_error);
    }
    super.checkCanUse();
  }
}
