package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A base class for simplifying classifiers. */
public abstract class SimplifyingClassifier extends Classifier {

  /** the selected attribute indexes */
  protected final int[] m_selectedAttributes;

  /**
   * create
   *
   * @param selectedAttributes
   *          the selected attributes
   */
  protected SimplifyingClassifier(final int[] selectedAttributes) {
    super();
    this.m_selectedAttributes = selectedAttributes;
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

  /** {@inheritDoc} */
  @Override
  public void render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    this.printDescription(textOutput, ETextCase.AT_SENTENCE_START);
  }
}
