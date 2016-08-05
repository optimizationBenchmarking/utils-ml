package weka.classifiers.trees;

import org.optimizationBenchmarking.utils.document.impl.SemanticComponentUtils;
import org.optimizationBenchmarking.utils.document.spec.ELabelType;
import org.optimizationBenchmarking.utils.document.spec.ICode;
import org.optimizationBenchmarking.utils.document.spec.ILabel;
import org.optimizationBenchmarking.utils.document.spec.ISectionBody;
import org.optimizationBenchmarking.utils.document.spec.ISemanticComponent;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.classifiers.trees.j48.WekaClassifierTreeAccessor;

/**
 * This class exists as cheap work-around for accessing the internal
 * variables of Weka's classifier trees.
 */
public final class WekaJ48Accessor {

  /**
   * Render the classifier tree to a given text output destination.
   *
   * @param name
   *          the classifier's name
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
      final ISemanticComponent name, final J48 tree,
      final IClassifierParameterRenderer renderer,
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
          WekaJ48Accessor.__renderJ48Classifier(tree, renderer, codeBody);
        }
      }
    } else {
      textOutput.append("Below you can find the structure of the ");//$NON-NLS-1$
      SemanticComponentUtils.printLongAndShortNameIfDifferent(name,
          textOutput, ETextCase.IN_SENTENCE);
      textOutput.append('.');
      textOutput.appendLineBreak();
      WekaJ48Accessor.__renderJ48Classifier(tree, renderer, textOutput);
    }
  }

  /**
   * Render the J48 classifier tree to a given text output destination.
   *
   * @param tree
   *          the tree to render
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  private static final void __renderJ48Classifier(final J48 tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaClassifierTreeAccessor.renderClassifierTree(tree.m_root, renderer,
        textOutput, 0);
  }
}
