package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A base class for simplifying classifiers. */
public abstract class SimplifyingClassifier extends Classifier {

  /** the selected feature indexes */
  protected final int[] m_selectedFeatures;

  /**
   * create
   *
   * @param selectedFeatures
   *          the selected features
   */
  protected SimplifyingClassifier(final int[] selectedFeatures) {
    super();
    if ((selectedFeatures == null) || (selectedFeatures.length <= 0)) {
      throw new IllegalArgumentException(//
          "Selected features cannot be null or empty."); //$NON-NLS-1$
    }
    this.m_selectedFeatures = selectedFeatures;
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
}
