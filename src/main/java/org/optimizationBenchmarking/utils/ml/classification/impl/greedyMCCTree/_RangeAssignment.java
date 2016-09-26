package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** the assignment of a value list or range to a classifier */
final class _RangeAssignment extends _Assignment {

  /** the values */
  private final double[][] m_values;

  /**
   * create the assignment
   *
   * @param values
   *          the values
   */
  _RangeAssignment(final double[][] values) {
    super();
    this.m_values = values;
  }

  /** {@inheritDoc} */
  @Override
  final boolean _check(final double value) {
    final boolean isNaN;

    isNaN = EFeatureType.featureDoubleIsUnspecified(value);
    for (final double[] check : this.m_values) {
      if (check == null) {
        if (isNaN) {
          return true;
        }
        continue;
      }
      if (_RangeAssignment._check(check[0], check[1], value, isNaN)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check whether a given value is in the range
   *
   * @param rangeMin
   *          the minimum of the range
   * @param rangeMax
   *          the maximum of the range
   * @param value
   *          the value
   * @param valueIsNaN
   *          is the value undefined
   * @return {@code true} if the value is in the range, {@code false}
   *         otherwise
   */
  static final boolean _check(final double rangeMin, final double rangeMax,
      final double value, final boolean valueIsNaN) {
    return ((((rangeMin <= Double.NEGATIVE_INFINITY) && (!valueIsNaN))
        || (rangeMin <= value))
        && (((rangeMax >= Double.POSITIVE_INFINITY) && (!valueIsNaN))
            || (value < rangeMax)));
  }

  /**
   * Check whether a given value is in the range
   *
   * @param rangeMin
   *          the minimum of the range
   * @param rangeMax
   *          the maximum of the range
   * @param value
   *          the value
   * @return {@code true} if the value is in the range, {@code false}
   *         otherwise
   */
  static final boolean _check(final double rangeMin, final double rangeMax,
      final double value) {
    return _RangeAssignment._check(rangeMin, rangeMax, value,
        EFeatureType.featureDoubleIsUnspecified(value));
  }

  /** {@inheritDoc} */
  @Override
  final void _render(final int attribute,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    boolean first, braces, has;

    first = true;
    braces = this.m_values.length > 1;
    for (final double[] values : this.m_values) {
      if (first) {
        first = false;
      } else {
        textOutput.append(" or ");//$NON-NLS-1$
      }

      if (braces) {
        textOutput.append('(');
      }

      if (values == null) {
        renderer.renderShortFeatureName(attribute, textOutput);
        textOutput.append(" is ");//$NON-NLS-1$
        textOutput.append(ClassificationTools.FEATURE_IS_UNSPECIFIED);
      } else {
        has = true;
        if (values[0] >= values[1]) {
          renderer.renderShortFeatureName(attribute, textOutput);
          textOutput.append(" = ");//$NON-NLS-1$
          renderer.renderFeatureValue(attribute, values[0], textOutput);
          if (braces) {
            textOutput.append(')');
          }
          continue;
        }
        if (values[0] > Double.NEGATIVE_INFINITY) {
          renderer.renderFeatureValue(attribute, values[0], textOutput);
          textOutput.append(" <= ");//$NON-NLS-1$
          has = false;
        }
        renderer.renderShortFeatureName(attribute, textOutput);
        if (values[1] < Double.POSITIVE_INFINITY) {
          textOutput.append(" < ");//$NON-NLS-1$
          renderer.renderFeatureValue(attribute, values[1], textOutput);
          if (braces) {
            textOutput.append(')');
          }
          continue;
        }
        if (has) {
          textOutput.append(" is ");//$NON-NLS-1$
          textOutput.append(ClassificationTools.FEATURE_IS_SPECIFIED);
        }
      }

      if (braces) {
        textOutput.append(')');
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  final double _complexity() {
    double sum;
    boolean first, has;

    first = true;
    sum = super._complexity();
    for (final double[] values : this.m_values) {
      if (first) {
        first = false;
      } else {
        sum += ClassificationTools.COMPLEXITY_LOGIC_UNIT;
      }

      if (values == null) {
        sum += (ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_FEATURE_UNIT);
        continue;
      }

      has = true;
      if (values[0] >= values[1]) {
        sum += (ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT);
        continue;
      }

      if (values[0] > Double.NEGATIVE_INFINITY) {
        sum += (ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT);
        has = false;
      }
      if (values[1] < Double.POSITIVE_INFINITY) {
        sum += (ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT);
        continue;
      }
      if (has) {
        sum += (ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_FEATURE_UNIT);
      }
    }
    return sum;
  }
}
