package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.document.impl.SemanticComponentUtils;
import org.optimizationBenchmarking.utils.document.spec.ELabelType;
import org.optimizationBenchmarking.utils.document.spec.ICode;
import org.optimizationBenchmarking.utils.document.spec.ILabel;
import org.optimizationBenchmarking.utils.document.spec.ISectionBody;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A base class for classifiers. */
public abstract class Classifier implements IClassifier {

  /** create */
  protected Classifier() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWord(this.getClass().getSimpleName(), textOut);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    return this.printShortName(textOut, textCase);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    return this.printLongName(textOut, textCase);
  }

  /** {@inheritDoc} */
  @Override
  public String getPathComponentSuggestion() {
    return this.getClass().getSimpleName();
  }

  /**
   * Render this classifier as code.
   *
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output
   */
  protected void renderAsCode(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    this.printDescription(textOutput, ETextCase.AT_SENTENCE_START);
  }

  /** {@inheritDoc} */
  @Override
  public void render(final IClassifierParameterRenderer renderer,
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
          SemanticComponentUtils.printLongAndShortNameIfDifferent(this,
              caption, ETextCase.IN_SENTENCE);
          caption.append('.');
        }
        try (final IText codeBody = code.body()) {
          this.renderAsCode(renderer, codeBody);
        }
      }
    } else {
      textOutput.append("Below you can find the structure of the ");//$NON-NLS-1$
      SemanticComponentUtils.printLongAndShortNameIfDifferent(this,
          textOutput, ETextCase.IN_SENTENCE);
      textOutput.append('.');
      textOutput.appendLineBreak();
      this.renderAsCode(renderer, textOutput);
    }
  }

}
