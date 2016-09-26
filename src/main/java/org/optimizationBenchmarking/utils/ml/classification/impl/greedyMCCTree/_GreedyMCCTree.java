package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.Classifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** a node in the tree-based classifier */
abstract class _GreedyMCCTree extends Classifier {

  /** the tree classifier node */
  _GreedyMCCTree() {
    super();
  }

  /**
   * render with a certain depth
   *
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output
   * @param depth
   *          the depth
   * @param isNewLine
   *          are we at a new line
   */
  abstract void _render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth,
      final boolean isNewLine);

  /** {@inheritDoc} */
  @Override
  public final void render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    this._render(renderer, textOutput, 0, true);
  }

  /**
   * Get the complexity of the expression
   *
   * @return the complexity of the expression
   */
  abstract double _complexity();

  /** {@inheritDoc} */
  @Override
  public ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWords(GreedyMCCTreeTrainer.NAME, textOut);
  }

  /** {@inheritDoc} */
  @Override
  public String getPathComponentSuggestion() {
    return "greedyMCCtree"; //$NON-NLS-1$
  }
}
