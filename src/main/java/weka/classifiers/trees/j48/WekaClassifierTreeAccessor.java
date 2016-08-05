package weka.classifiers.trees.j48;

import org.optimizationBenchmarking.utils.document.spec.ELabelType;
import org.optimizationBenchmarking.utils.document.spec.ICode;
import org.optimizationBenchmarking.utils.document.spec.ILabel;
import org.optimizationBenchmarking.utils.document.spec.ISectionBody;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;
import org.optimizationBenchmarking.utils.text.tokenizers.LineIterator;

import weka.classifiers.trees.J48;

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
   *          the text output destination (could be an instance of
   *          {@link org.optimizationBenchmarking.utils.document.spec.IComplexText}
   *          or even
   *          {@link org.optimizationBenchmarking.utils.document.spec.ISectionBody}
   *          , in which case you should do something cool...)
   */
  public static final void renderJ48Classifier(final J48 tree,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    final ILabel label;
    String src;
    ISectionBody body;
    boolean notFirst;

    try {
      src = tree.toSource("classifier").toString(); //$NON-NLS-1$
    } catch (@SuppressWarnings("unused") final Throwable error) {
      src = "source"; //$NON-NLS-1$
    }

    if (textOutput instanceof ISectionBody) {
      body = ((ISectionBody) textOutput);
      label = body.createLabel(ELabelType.CODE);
      body.append("The classifier is rendered in "); //$NON-NLS-1$
      body.reference(ETextCase.IN_SENTENCE, ESequenceMode.AND, label);
      body.append('.');

      try (final ICode code = ((ISectionBody) textOutput).code(label,
          true)) {
        try (final IText caption = code.caption()) {
          caption.append("caption");//$NON-NLS-1$
        }
        notFirst = true;
        try (final IText codeBody = code.body()) {
          for (final String line : new LineIterator(src, false, false)) {
            if (notFirst) {
              notFirst = false;
            } else {
              codeBody.appendLineBreak();
            }
            codeBody.append(line);
          }
        }
      }
    } else {
      textOutput.append(src);
    }

  }
}
