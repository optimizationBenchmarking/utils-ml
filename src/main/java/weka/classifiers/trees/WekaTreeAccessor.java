package weka.classifiers.trees;

import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.document.impl.SemanticComponentUtils;
import org.optimizationBenchmarking.utils.document.spec.ELabelType;
import org.optimizationBenchmarking.utils.document.spec.ICode;
import org.optimizationBenchmarking.utils.document.spec.ILabel;
import org.optimizationBenchmarking.utils.document.spec.ISectionBody;
import org.optimizationBenchmarking.utils.document.spec.ISemanticComponent;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
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
    final ILabel label;
    ISectionBody body;
    if (textOutput instanceof ISectionBody) {
      body = ((ISectionBody) textOutput);
      label = body.createLabel(ELabelType.CODE);
      body.append("The structure of the classifier is rendered in "); //$NON-NLS-1$
      body.reference(ETextCase.IN_SENTENCE, ESequenceMode.AND, label);
      body.append('.');

      try (final ICode code = ((ISectionBody) textOutput).code(label,
          true)) {
        try (final IText caption = code.caption()) {
          caption.append("The structure of the ");//$NON-NLS-1$
          SemanticComponentUtils.printLongAndShortNameIfDifferent(name,
              caption, ETextCase.IN_SENTENCE);
          caption.append('.');
        }
        try (final IText codeBody = code.body()) {
          WekaTreeAccessor.__renderJ48Classifier(selectedFeatures, tree,
              renderer, codeBody);
        }
      }
    } else {
      textOutput.append("Below you can find the structure of the ");//$NON-NLS-1$
      SemanticComponentUtils.printLongAndShortNameIfDifferent(name,
          textOutput, ETextCase.IN_SENTENCE);
      textOutput.append('.');
      textOutput.appendLineBreak();
      WekaTreeAccessor.__renderJ48Classifier(selectedFeatures, tree,
          renderer, textOutput);
    }
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
    final ILabel label;
    ISectionBody body;
    if (textOutput instanceof ISectionBody) {
      body = ((ISectionBody) textOutput);
      label = body.createLabel(ELabelType.CODE);
      body.append("The structure of the classifier is rendered in "); //$NON-NLS-1$
      body.reference(ETextCase.IN_SENTENCE, ESequenceMode.AND, label);
      body.append('.');

      try (final ICode code = ((ISectionBody) textOutput).code(label,
          true)) {
        try (final IText caption = code.caption()) {
          caption.append("The structure of the ");//$NON-NLS-1$
          SemanticComponentUtils.printLongAndShortNameIfDifferent(name,
              caption, ETextCase.IN_SENTENCE);
          caption.append('.');
        }
        try (final IText codeBody = code.body()) {
          WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures,
              tree, renderer, codeBody);
        }
      }
    } else {
      textOutput.append("Below you can find the structure of the ");//$NON-NLS-1$
      SemanticComponentUtils.printLongAndShortNameIfDifferent(name,
          textOutput, ETextCase.IN_SENTENCE);
      textOutput.append('.');
      textOutput.appendLineBreak();
      WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures, tree,
          renderer, textOutput);
    }
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
              ? ClassificationTools.doubleToClass(currentProbs[0])//
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
        ClassificationTools.printFeatureExpression(//
            selectedFeatures[tree.m_Attribute], //
            EComparison.EQUAL, index, renderer, textOutput);
        textOutput.append(ClassificationTools.RULE_THEN);
      } else {
        if (index <= 0) {
          textOutput.append(ClassificationTools.RULE_IF);
          ClassificationTools.printFeatureExpression(//
              selectedFeatures[tree.m_Attribute], EComparison.LESS,
              tree.m_SplitPoint, renderer, textOutput);
          textOutput.append(ClassificationTools.RULE_THEN);
        } else {
          textOutput.append(ClassificationTools.RULE_ELSE);
        }
      }

      if (tree.m_Successors[index].m_Attribute < 0) {
        textOutput.append(' ');

        if ((successorProbs = tree.m_ClassProbs) == null) {
          successorProbs = tree.m_ClassProbs;
        }

        ClassificationTools
            .printClass(//
                ((tree.m_Info.classAttribute().isNumeric())//
                    ? ClassificationTools.doubleToClass(successorProbs[0])//
                    : Utils.maxIndex(successorProbs)),
                renderer, textOutput);

      } else {
        WekaTreeAccessor.__renderREPTreeClassifier(selectedFeatures,
            tree.m_Successors[index], tree, renderer, textOutput,
            (depth + 2), false);
      }
    }
  }
}