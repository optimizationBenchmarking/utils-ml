package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** the assignment of a value list or range to a classifier */
class _Assignment {
  /** the corresponding classifier */
  _GreedyMCCTree m_classifier;

  /** create the assignment */
  _Assignment() {
    super();
  }

  /**
   * check if the assignment fits
   *
   * @param value
   *          the value
   * @return {@code true} if the assignment fits, {@code false} otherwise
   */
  boolean _check(final double value) {
    return true;
  }

  /**
   * render this classifier's expression
   *
   * @param attribute
   *          the attribute
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the output
   */
  void _render(final int attribute,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append(ClassificationTools.RULE_ALWAYS_TRUE);
  }

  /**
   * Get the complexity of the expression
   *
   * @return the complexity of the expression
   */
  double _complexity() {
    return 0d;
  }
}
