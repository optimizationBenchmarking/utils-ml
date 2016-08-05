package weka.classifiers.trees.j48;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * This class exists as cheap work-around for accessing the internal
 * variables of Weka's classifier trees.
 */
public final class WekaClassifierTreeAccessor {

  /**
   * Render the classifier tree to a given text output destination.
   *
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   * @param depth
   *          the current depth
   */
  public static final void renderClassifierTree(final ClassifierTree tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth) {
    if (tree.m_isLeaf) {
      textOutput.append("class = "); //$NON-NLS-1$
      renderer.renderShortClassName(
          tree.m_localModel.distribution().maxClass(0), textOutput);
      return;
    }
  }
}
