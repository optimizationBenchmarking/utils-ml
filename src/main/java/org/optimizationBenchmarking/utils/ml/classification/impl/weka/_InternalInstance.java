package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;

import weka.core.DenseInstance;

/** the internal instance class */
final class _InternalInstance extends DenseInstance {

  /** the maximum allowed value */
  private static final double FEATURE_MAX = Double.MAX_VALUE;
  /** the minimum allowed value */
  private static final double FEATURE_MIN = (-_InternalInstance.FEATURE_MAX);

  /** the serial version UID */
  private static final long serialVersionUID = 1L;

  /**
   * create
   *
   * @param sample
   *          the classified sample
   * @param selection
   *          the selection
   */
  _InternalInstance(final ClassifiedSample sample, final int[] selection) {
    super(1, new double[selection.length + 1]);
    this._assign(sample.featureValues, selection);
    this.m_AttValues[selection.length] = sample.sampleClass;
  }

  /**
   * create
   *
   * @param length
   *          the length of the vector
   */
  _InternalInstance(final int length) {
    super(1, new double[length]);
  }

  /** {@inheritDoc} */
  @Override
  public final _InternalInstance copy() {
    return this;
  }

  /**
   * format a feature value for assignment
   *
   * @param value
   *          the feature value
   * @return the assignment
   */
  private static final double __format(final double value) {
    return ((value <= _InternalInstance.FEATURE_MIN)
        ? _InternalInstance.FEATURE_MIN
        : ((value >= _InternalInstance.FEATURE_MAX)
            ? _InternalInstance.FEATURE_MAX : value));
  }

  /**
   * copy the given data into this array
   *
   * @param data
   *          the data to copy
   * @param selection
   *          the selection
   */
  final void _assign(final double[] data, final int[] selection) {
    int index;
    index = (-1);
    for (final int chosen : selection) {
      this.m_AttValues[++index] = _InternalInstance.__format(data[chosen]);
    }
  }
}
