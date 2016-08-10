package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;

import weka.core.DenseInstance;

/** the internal instance class */
final class _InternalInstance extends DenseInstance {

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
    int index;
    index = (-1);
    for (final int chosen : selection) {
      this.m_AttValues[++index] = sample.featureValues[chosen];
    }
    this.m_AttValues[++index] = sample.sampleClass;
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
   * copy the given data into this array
   *
   * @param data
   *          the data to copy
   * @param selection
   *          the selection
   */
  final void _assign(final double[] data, final int[] selection) {
    final int length;
    int index;

    if ((length = this.m_AttValues.length) >= selection.length) {
      System.arraycopy(data, 0, this.m_AttValues, 0, length);
    } else {
      index = (-1);
      for (final int chosen : selection) {
        this.m_AttValues[++index] = data[chosen];
      }
    }
  }
}
