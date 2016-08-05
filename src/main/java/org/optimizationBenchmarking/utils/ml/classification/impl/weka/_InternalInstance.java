package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import weka.core.DenseInstance;

/** the internal instance class */
final class _InternalInstance extends DenseInstance {

  /** the serial version UID */
  private static final long serialVersionUID = 1L;

  /**
   * create
   *
   * @param attValues
   *          the attribute values
   */
  _InternalInstance(final double[] attValues) {
    super(1, attValues);
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
   */
  final void _assign(final double[] data) {
    System.arraycopy(data, 0, this.m_AttValues, 0,
        this.m_AttValues.length);
  }
}
