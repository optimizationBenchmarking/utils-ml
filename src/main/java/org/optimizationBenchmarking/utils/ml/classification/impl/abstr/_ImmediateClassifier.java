package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

/** a trivial, immediate classifier result */
abstract class _ImmediateClassifier extends Classifier
    implements IClassifierTrainingJob, IClassifierTrainingResult {

  /** create the immediate classifier */
  _ImmediateClassifier() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifierTrainingResult call() {
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifier getClassifier() {
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public final double getQuality() {
    return 0;
  }
}
