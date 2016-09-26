package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.TextUtils;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** the decision node of the tree classifier */
final class _GreedyMCCTreeDecisionNode extends _GreedyMCCTree {

  /** the attribute */
  private final int m_attribute;

  /** the assignments */
  private final _Assignment[] m_assignments;

  /**
   * Create the tree-based classifier
   *
   * @param attribute
   *          the attribute
   * @param assignments
   *          the assignments
   */
  _GreedyMCCTreeDecisionNode(final int attribute,
      final _Assignment[] assignments) {
    super();
    this.m_attribute = attribute;
    this.m_assignments = assignments;
  }

  /** {@inheritDoc} */
  @Override
  public final int classify(final double[] features) {
    for (final _Assignment assignment : this.m_assignments) {
      if (assignment._check(features[this.m_attribute])) {
        return assignment.m_classifier.classify(features);
      }
    }

    return this.m_assignments[this.m_assignments.length - 1].m_classifier
        .classify(features);
  }

  /** {@inheritDoc} */
  @Override
  final void _render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth,
      final boolean isNewLine) {
    int index, max;
    boolean needsNewLine;

    needsNewLine = !isNewLine;

    max = (this.m_assignments.length - 1);
    for (index = 0; index <= max; index++) {
      if (needsNewLine) {
        textOutput.appendLineBreak();
      } else {
        needsNewLine = true;
      }
      TextUtils.appendNonBreakingSpaces(depth << 1, textOutput);
      textOutput.append((index <= 0) ? ClassificationTools.RULE_IF
          : ((index < max) ? ClassificationTools.RULE_ELSE_IF
              : ClassificationTools.RULE_ELSE));
      if (index < max) {
        this.m_assignments[index]._render(this.m_attribute, renderer,
            textOutput);
        textOutput.append(ClassificationTools.RULE_THEN);
      }
      this.m_assignments[index].m_classifier._render(renderer, textOutput,
          depth + 1, false);
    }
  }

  /** {@inheritDoc} */
  @Override
  final double _complexity() {
    double[] values;
    int assignmentIndex, index, max;

    max = this.m_assignments.length;
    values = new double[max << 1];
    --max;
    for (assignmentIndex = 0, index = (-1); assignmentIndex <= max; ++assignmentIndex) {
      values[++index] = this.m_assignments[assignmentIndex].m_classifier
          ._complexity();
      values[++index] = ClassificationTools.COMPLEXITY_DECISION_UNIT;
      if (assignmentIndex < max) {
        values[index] += this.m_assignments[assignmentIndex]._complexity();
      }
    }

    return ClassificationTools.complexityNested(values);
  }
}
