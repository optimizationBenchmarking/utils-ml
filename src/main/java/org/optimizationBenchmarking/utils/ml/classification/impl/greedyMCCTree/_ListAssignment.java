package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** the assignment of a value list or range to a classifier */
final class _ListAssignment extends _Assignment {

  /** the values */
  final int[] m_values;

  /**
   * create the assignment
   *
   * @param values
   *          the values
   */
  _ListAssignment(final int[] values) {
    super();
    this.m_values = values;
  }

  /** {@inheritDoc} */
  @Override
  final boolean _check(final double value) {
    final int intVal;

    intVal = EFeatureType.featureDoubleToNominal(value);
    for (final int check : this.m_values) {
      if (check == intVal) {
        return true;
      }
    }
    return false;
  }

  /** {@inheritDoc} */
  @Override
  final void _render(final int attribute,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    boolean first;

    renderer.renderShortFeatureName(attribute, textOutput);
    if (this.m_values.length <= 1) {
      textOutput.append(" = "); //$NON-NLS-1$
    } else {
      textOutput.append(" in {"); //$NON-NLS-1$
    }

    first = true;
    for (final int value : this.m_values) {
      if (first) {
        first = false;
      } else {
        textOutput.append(", ");//$NON-NLS-1$
      }
      if (EFeatureType.featureNominalIsUnspecified(value)) {
        textOutput.append(ClassificationTools.FEATURE_IS_UNSPECIFIED);
      } else {
        renderer.renderFeatureValue(attribute, value, textOutput);
      }
    }
    if (this.m_values.length > 1) {
      textOutput.append('}');
    }
  }

  /** {@inheritDoc} */
  @Override
  final double _complexity() {
    return ((ClassificationTools.COMPLEXITY_COMPARISON_UNIT
        + ClassificationTools.COMPLEXITY_FEATURE_UNIT
        + ClassificationTools.COMPLEXITY_CONSTANT_UNIT)
        * this.m_values.length) + super._complexity();
  }
}
