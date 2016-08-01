package weka.classifiers.trees.j48;

import org.optimizationBenchmarking.utils.document.spec.ICode;
import org.optimizationBenchmarking.utils.document.spec.ISectionBody;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
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
    String src;
    boolean notFirst;

    try {
      src = tree.toSource("classifier").toString(); //$NON-NLS-1$
    } catch (@SuppressWarnings("unused") final Throwable error) {
      src = "source"; //$NON-NLS-1$
    }

    if (textOutput instanceof ISectionBody) {
      try (final ICode code = ((ISectionBody) textOutput).code(null,
          true)) {
        try (final IText caption = code.caption()) {
          caption.append("caption");//$NON-NLS-1$
        }
        notFirst = true;
        try (final IText body = code.body()) {
          for (final String line : new LineIterator(src, false, false)) {
            if (notFirst) {
              notFirst = true;
            } else {
              body.appendLineBreak();
            }
            body.append(line);
          }
        }
      }
    } else {
      textOutput.append(src);
    }

  }
}
