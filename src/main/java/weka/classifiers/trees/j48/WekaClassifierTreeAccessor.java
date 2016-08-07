package weka.classifiers.trees.j48;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.TextUtils;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.core.Instances;

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
    WekaClassifierTreeAccessor.__renderClassifierTree(tree, renderer,
        textOutput, depth, true);
  }

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
   * @param isNewLine
   *          is the current line new?
   */
  private static final void __renderClassifierTree(
      final ClassifierTree tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth,
      final boolean isNewLine) {
    final int end;
    int index;

    if (tree.m_isLeaf) {
      if (isNewLine) {
        TextUtils.appendNonBreakingSpaces(depth, textOutput);
      } else {
        textOutput.append(' ');
      }
      textOutput.append("class = "); //$NON-NLS-1$
      renderer.renderShortClassName(
          tree.m_localModel.distribution().maxClass(0), textOutput);
      return;
    }

    end = tree.m_sons.length - 1;
    for (index = 0; index <= end; index++) {
      if ((index > 0) || (!isNewLine)) {
        textOutput.appendLineBreak();
      }
      TextUtils.appendNonBreakingSpaces(depth, textOutput);

      textOutput.append(((index <= 0) ? "if " : ((index < end) //$NON-NLS-1$
          ? "else if " : "else ")));//$NON-NLS-1$//$NON-NLS-2$
      WekaClassifierTreeAccessor.__renderExpression(tree.m_localModel,
          index, tree.m_train, renderer, textOutput);
      if (index < end) {
        textOutput.append(" then");//$NON-NLS-1$
      }
      if (tree.m_sons[index].m_isLeaf) {
        textOutput.append(" class = "); //$NON-NLS-1$
        renderer.renderShortClassName(
            tree.m_localModel.distribution().maxClass(index), textOutput);
      } else {
        WekaClassifierTreeAccessor.__renderClassifierTree(
            tree.m_sons[index], renderer, textOutput, (depth + 2), false);
      }
    }
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(
      final ClassifierSplitModel model, final int index,
      final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    if (model instanceof BinC45Split) {
      WekaClassifierTreeAccessor.__renderExpression(((BinC45Split) model),
          index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof C45Split) {
      WekaClassifierTreeAccessor.__renderExpression(((C45Split) model),
          index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof NoSplit) {
      WekaClassifierTreeAccessor.__renderExpression(((NoSplit) model),
          index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof NBTreeSplit) {
      WekaClassifierTreeAccessor.__renderExpression(((NBTreeSplit) model),
          index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof NBTreeNoSplit) {
      WekaClassifierTreeAccessor.__renderExpression(
          ((NBTreeNoSplit) model), index, trainingData, renderer,
          textOutput);
      return;
    }
    throw new IllegalArgumentException(
        "Cannot deal with split " + TextUtils.className(model) + '.'); //$NON-NLS-1$
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(final BinC45Split model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    renderer.renderShortFeatureName(model.m_attIndex, textOutput);
    textOutput.append(' ');

    if (trainingData.attribute(model.m_attIndex).isNominal()) {
      if (index <= 0) {
        textOutput.append('=');
      } else {
        textOutput.append('!');
        textOutput.append('=');
      }
    } else {
      if (index <= 0) {
        textOutput.append('<');
        textOutput.append('=');
      } else {
        textOutput.append('>');
      }
    }

    textOutput.append(' ');
    renderer.renderFeatureValue(model.m_attIndex, model.m_splitPoint,
        textOutput);
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(final C45Split model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    renderer.renderShortFeatureName(model.m_attIndex, textOutput);
    textOutput.append(' ');

    if (trainingData.attribute(model.m_attIndex).isNominal()) {
      textOutput.append('=');
    } else {
      if (index <= 0) {
        textOutput.append('<');
        textOutput.append('=');
      } else {
        textOutput.append('>');
      }
    }

    textOutput.append(' ');
    renderer.renderFeatureValue(model.m_attIndex, model.m_splitPoint,
        textOutput);
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(final NoSplit model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append("true"); //$NON-NLS-1$
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(final NBTreeSplit model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaClassifierTreeAccessor.__renderExpression(model.m_c45S, index,
        trainingData, renderer, textOutput);
  }

  /**
   * render the expression
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @param trainingData
   *          the training data
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the destination
   */
  private static final void __renderExpression(final NBTreeNoSplit model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append("true"); //$NON-NLS-1$
  }
}
