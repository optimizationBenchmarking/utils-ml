package weka.classifiers.trees;

import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.document.spec.ISemanticComponent;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.TextUtils;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.classifiers.trees.j48.WekaClassifierTreeAccessor;
import weka.core.Utils;

/**
 * This class exists as cheap work-around for accessing the internal
 * variables of Weka's classifier trees.
 */
public final class WekaTreeAccessor {

  /**
   * Print an expression which compares a the value of an feature with a
   * specified value.
   *
   * @param feature
   *          the feature
   * @param comparison
   *          the comparison
   * @param value
   *          the value to compare with
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  public static final void printFeatureExpression(final int feature,
      final EComparison comparison, final double value,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    ClassificationTools.printFeatureExpression(feature, comparison, //
        ((value != value) || (value <= (-Double.MAX_VALUE))
            || (value >= Double.MAX_VALUE))//
                ? EFeatureType.UNSPECIFIED_DOUBLE : value, //
        renderer, textOutput);
  }

  /**
   * Render the J48 classifier tree to a given text output destination.
   *
   * @param name
   *          the classifier's name
   * @param selectedFeatures
   *          the selected features
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination (could be an instance of
   *          {@link org.optimizationBenchmarking.utils.document.spec.IComplexText}
   *          or even
   *          {@link org.optimizationBenchmarking.utils.document.spec.ISectionBody}
   *          , in which case you should do something cool...)
   */
  public static final void renderJ48Classifier(
      final ISemanticComponent name, final int[] selectedFeatures,
      final J48 tree, final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaTreeAccessor.__renderJ48Classifier(selectedFeatures, tree,
        renderer, textOutput);
  }

  /**
   * Render the J48 classifier tree to a given text output destination.
   *
   * @param selectedFeatures
   *          the selected features
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  private static final void __renderJ48Classifier(
      final int[] selectedFeatures, final J48 tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaClassifierTreeAccessor.renderClassifierTree(selectedFeatures,
        tree.m_root, renderer, textOutput, 0);
  }

  /**
   * Get the complexity of a J48 tree
   *
   * @param tree
   *          the tree
   * @return its complexity
   */
  public static final double getJ48Complexity(final J48 tree) {
    return WekaClassifierTreeAccessor
        .getClassifierTreeComplexity(tree.m_root);
  }

  /**
   * Render the classifier REPTree tree to a given text output destination.
   *
   * @param name
   *          the classifier's name
   * @param selectedFeatures
   *          the selected features
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination (could be an instance of
   *          {@link org.optimizationBenchmarking.utils.document.spec.IComplexText}
   *          or even
   *          {@link org.optimizationBenchmarking.utils.document.spec.ISectionBody}
   *          , in which case you should do something cool...)
   */
  public static final void renderREPTreeClassifier(
      final ISemanticComponent name, final int[] selectedFeatures,
      final REPTree tree, final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures, tree,
        renderer, textOutput);
  }

  /**
   * Render the REPTree classifier tree to a given text output destination.
   *
   * @param selectedFeatures
   *          the selected features
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  private static final void __renderREPTreeClassifier(
      final int[] selectedFeatures, final REPTree tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures,
        tree.m_Tree, tree.m_Tree, renderer, textOutput, 0, true);
  }

  /**
   * Render the REPTree classifier tree to a given text output destination.
   *
   * @param selectedFeatures
   *          the selected features
   * @param tree
   *          the tree to render
   * @param parent
   *          the parent tree
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   * @param depth
   *          the current depth
   * @param isNewLine
   *          is the current line new?
   */
  private static final void __renderREPTreeClassifier(
      final int[] selectedFeatures, final REPTree.Tree tree,
      final REPTree.Tree parent,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput, final int depth,
      final boolean isNewLine) {
    final double[] currentProbs;
    final int end;
    double[] successorProbs;
    int index;

    if (tree.m_ClassProbs == null) {
      currentProbs = parent.m_ClassProbs;
    } else {
      currentProbs = tree.m_ClassProbs;
    }

    if (tree.m_Attribute < 0) {
      if (isNewLine) {
        TextUtils.appendNonBreakingSpaces(depth, textOutput);
      } else {
        textOutput.append(' ');
      }
      ClassificationTools.printClass(//
          ((tree.m_Info.classAttribute().isNumeric())//
              ? EFeatureType.doubleToClass(currentProbs[0])//
              : Utils.maxIndex(currentProbs)),
          renderer, textOutput);
      return;
    }

    end = tree.m_Successors.length - 1;
    for (index = 0; index <= end; index++) {
      if ((index > 0) || (!isNewLine)) {
        textOutput.appendLineBreak();
      }
      TextUtils.appendNonBreakingSpaces(depth, textOutput);

      if (tree.m_Info.attribute(tree.m_Attribute).isNominal()) {
        textOutput.append(((index <= 0) ? ClassificationTools.RULE_IF
            : ClassificationTools.RULE_ELSE_IF));
        WekaTreeAccessor.printFeatureExpression(//
            selectedFeatures[tree.m_Attribute], //
            EComparison.EQUAL, index, renderer, textOutput);
        textOutput.append(ClassificationTools.RULE_THEN);
      } else {
        if (index <= 0) {
          textOutput.append(ClassificationTools.RULE_IF);
          WekaTreeAccessor.printFeatureExpression(//
              selectedFeatures[tree.m_Attribute], EComparison.LESS,
              tree.m_SplitPoint, renderer, textOutput);
          textOutput.append(ClassificationTools.RULE_THEN);
        } else {
          textOutput.append(ClassificationTools.RULE_ELSE);
        }
      }

      if (tree.m_Successors[index].m_Attribute < 0) {
        textOutput.append(' ');

        successorProbs = tree.m_Successors[index].m_ClassProbs;
        if (successorProbs == null) {
          successorProbs = tree.m_ClassProbs;
        }

        ClassificationTools
            .printClass(//
                ((tree.m_Info.classAttribute().isNumeric())//
                    ? EFeatureType.doubleToClass(successorProbs[0])//
                    : Utils.maxIndex(successorProbs)),
                renderer, textOutput);

      } else {
        WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures,
            tree.m_Successors[index], tree, renderer, textOutput,
            (depth + 2), false);
      }
    }
  }

  /**
   * Get the complexity of a REPTree tree classifier
   *
   * @param tree
   *          the tree to compute the complexity of
   * @return its complexity
   */
  public static final double getREPTreeComplexity(final REPTree tree) {
    return WekaTreeAccessor.__getREPTreeComplexity(tree.m_Tree);

  }

  /**
   * Get the complexity of a REPTree tree classifier
   *
   * @param tree
   *          the tree to compute the complexity of
   * @return its complexity
   */
  private static final double __getREPTreeComplexity(
      final REPTree.Tree tree) {
    int sonIndex, dataIndex;
    double[] add;
    final int end;

    if (tree.m_Attribute < 0) {
      return ClassificationTools.COMPLEXITY_CLASS_UNIT;
    }

    end = tree.m_Successors.length - 1;
    add = new double[tree.m_Successors.length << 1];
    for (sonIndex = 0, dataIndex = (-1); sonIndex <= end; sonIndex++) {

      if (tree.m_Info.attribute(tree.m_Attribute).isNominal()
          || (sonIndex <= 0)) {
        add[++dataIndex] = ClassificationTools.COMPLEXITY_DECISION_UNIT
            + ClassificationTools.COMPLEXITY_FEATURE_UNIT
            + ClassificationTools.COMPLEXITY_COMPARISON_UNIT
            + ClassificationTools.COMPLEXITY_CONSTANT_UNIT;
      } else {
        add[++dataIndex] = ClassificationTools.COMPLEXITY_DECISION_UNIT;
      }

      add[++dataIndex] = WekaTreeAccessor
          .__getREPTreeComplexity(tree.m_Successors[sonIndex]);

    }

    return ClassificationTools.complexityNested(add);
  }
}