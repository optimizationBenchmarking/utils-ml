package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** a leaf of the tree classifier */
final class _GreedyMCCTreeLeaf extends _GreedyMCCTree {
  /** the class */
  private final int m_class;

  /**
   * create
   *
   * @param clazz
   *          the class
   */
  _GreedyMCCTreeLeaf(final int clazz) {
    super();
    this.m_class = clazz;
  }

  /** {@inheritDoc} */
  @Override
  public final int classify(final double[] features) {
    return this.m_class;
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;
    next = textCase.appendWord("always", textOut);//$NON-NLS-1$
    textOut.append(' ');
    next = next.appendWord("class", textOut);//$NON-NLS-1$
    textOut.append(' ');
    textOut.append(this.m_class);
    return next;
  }

  /** {@inheritDoc} */
  @Override
  public String getPathComponentSuggestion() {
    return String.valueOf(this.m_class);
  }

  /** {@inheritDoc} */
  @Override
  final void _render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth,
      final boolean isNewLine) {
    renderer.renderShortClassName(this.m_class, textOutput);
  }

  /** {@inheritDoc} */
  @Override
  final double _complexity() {
    return ClassificationTools.COMPLEXITY_CLASS_UNIT;
  }
}
