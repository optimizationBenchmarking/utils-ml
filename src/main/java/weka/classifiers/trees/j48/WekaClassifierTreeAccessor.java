package weka.classifiers.trees.j48;

import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
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
   * @param selectedAttributes
   *          the selected attributes
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   * @param depth
   *          the current depth
   */
  public static final void renderClassifierTree(
      final int[] selectedAttributes, final ClassifierTree tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth) {
    WekaClassifierTreeAccessor.__renderClassifierTree(selectedAttributes,
        tree, renderer, textOutput, depth, true);
  }

  /**
   * Render the classifier tree to a given text output destination.
   *
   * @param selectedAttributes
   *          the selected attributes
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
      final int[] selectedAttributes, final ClassifierTree tree,
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
      ClassificationTools.printClass(
          tree.m_localModel.distribution().maxClass(0), renderer,
          textOutput);
      return;
    }

    end = tree.m_sons.length - 1;
    for (index = 0; index <= end; index++) {
      if ((index > 0) || (!isNewLine)) {
        textOutput.appendLineBreak();
      }
      TextUtils.appendNonBreakingSpaces(depth, textOutput);

      textOutput.append(((index <= 0) ? ClassificationTools.RULE_IF
          : ((index < end) ? ClassificationTools.RULE_ELSE_IF
              : ClassificationTools.RULE_ELSE)));
      if (index < end) {
        WekaClassifierTreeAccessor.__renderExpression(selectedAttributes,
            tree.m_localModel, index, tree.m_train, renderer, textOutput);
        textOutput.append(ClassificationTools.RULE_THEN);
      }
      if (tree.m_sons[index].m_isLeaf) {
        textOutput.append(' ');
        ClassificationTools.printClass(
            tree.m_localModel.distribution().maxClass(index), renderer,
            textOutput);
      } else {
        WekaClassifierTreeAccessor.__renderClassifierTree(
            selectedAttributes, tree.m_sons[index], renderer, textOutput,
            (depth + 2), false);
      }
    }
  }

  /**
   * render the expression
   *
   * @param selectedAttributes
   *          the selected attributes
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
      final int[] selectedAttributes, final ClassifierSplitModel model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    if (model instanceof BinC45Split) {
      WekaClassifierTreeAccessor.__renderExpression(selectedAttributes,
          ((BinC45Split) model), index, trainingData, renderer,
          textOutput);
      return;
    }
    if (model instanceof C45Split) {
      WekaClassifierTreeAccessor.__renderExpression(selectedAttributes,
          ((C45Split) model), index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof NoSplit) {
      WekaClassifierTreeAccessor.__renderExpression(((NoSplit) model),
          index, trainingData, renderer, textOutput);
      return;
    }
    if (model instanceof NBTreeSplit) {
      WekaClassifierTreeAccessor.__renderExpression(selectedAttributes,
          ((NBTreeSplit) model), index, trainingData, renderer,
          textOutput);
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
   * @param selectedAttributes
   *          the selected attributes
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
      final int[] selectedAttributes, final BinC45Split model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {

    if (trainingData.attribute(model.m_attIndex).isNominal()) {
      ClassificationTools.printFeatureExpression(
          selectedAttributes[model.m_attIndex],
          ((index <= 0) ? EComparison.EQUAL : EComparison.NOT_EQUAL),
          model.m_splitPoint, renderer, textOutput);
    } else {
      ClassificationTools
          .printFeatureExpression(selectedAttributes[model.m_attIndex],
              ((index <= 0) ? EComparison.LESS_OR_EQUAL
                  : EComparison.GREATER),
              model.m_splitPoint, renderer, textOutput);
    }
  }

  /**
   * render the expression
   *
   * @param selectedAttributes
   *          the selected attributes
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
      final int[] selectedAttributes, final C45Split model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {

    if (trainingData.attribute(model.m_attIndex).isNominal()) {
      ClassificationTools.printFeatureExpression(
          selectedAttributes[model.m_attIndex], EComparison.EQUAL,
          model.m_splitPoint, renderer, textOutput);
    } else {
      ClassificationTools
          .printFeatureExpression(selectedAttributes[model.m_attIndex],
              ((index <= 0) ? EComparison.LESS_OR_EQUAL
                  : EComparison.GREATER),
              model.m_splitPoint, renderer, textOutput);
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
  private static final void __renderExpression(final NoSplit model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append(ClassificationTools.RULE_ALWAYS_TRUE);
  }

  /**
   * render the expression
   *
   * @param selectedAttributes
   *          the selected attributes
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
      final int[] selectedAttributes, final NBTreeSplit model,
      final int index, final Instances trainingData,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaClassifierTreeAccessor.__renderExpression(selectedAttributes,
        model.m_c45S, index, trainingData, renderer, textOutput);
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
    textOutput.append(ClassificationTools.RULE_ALWAYS_TRUE);
  }
}
