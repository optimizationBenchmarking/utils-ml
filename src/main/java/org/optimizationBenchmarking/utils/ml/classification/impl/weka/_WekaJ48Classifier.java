package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibBook;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.WekaTreeAccessor;

/**
 * The wrapper for Weka's {@link weka.classifiers.trees.J48} classifiers
 * classifier.
 */
final class _WekaJ48Classifier extends _WekaClassifier<J48> {

  /** the book */
  private static final BibBook C45 = new BibBook(//
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("John Ross", "Quinlan") //$NON-NLS-1$//$NON-NLS-2$
  }), //
      "C4.5: Programs for Machine Learning", //$NON-NLS-1$
      new BibDate(1993), //
      BibAuthors.EMPTY_AUTHORS, //
      new BibOrganization(//
          "Morgan Kaufmann Publishers Inc.", //$NON-NLS-1$
          "San Francisco, CA, USA", null), //$NON-NLS-1$
      null, null, null, null, "1558602402", null, null//$NON-NLS-1$

  );

  /**
   * Create the Weka J48 classifier wrapper
   *
   * @param selectedFeatures
   *          the selected features
   * @param classifier
   *          the classifier
   * @param instance
   *          to use
   */
  _WekaJ48Classifier(final int[] selectedFeatures, final J48 classifier,
      final _InternalInstance instance) {
    super(selectedFeatures, classifier, instance);
  }

  /** {@inheritDoc} */
  @Override
  public final void render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaTreeAccessor.renderJ48Classifier(this, this.m_selectedFeatures,
        this.m_classifier, renderer, textOutput);
  }

  /**
   * print the description
   *
   * @param textOut
   *          the text output device
   * @param textCase
   *          the text case
   * @param addInfos
   *          can we add infos and potentially insert citations?
   * @return the next case
   */
  private final ETextCase __printDescription(final ITextOutput textOut,
      final ETextCase textCase, final boolean addInfos) {
    return _WekaJ48Classifier._printDescription(textOut, textCase,
        (this.m_classifier.getUnpruned()
            ? _WekaJ48ClassifierTrainingJob.PRUNING_OFF
            : (this.m_classifier.getReducedErrorPruning()
                ? _WekaJ48ClassifierTrainingJob.PRUNING_REDUCED_ERROR
                : _WekaJ48ClassifierTrainingJob.PRUNING_ON)),
        this.m_classifier.getBinarySplits(), addInfos);
  }

  /**
   * print the description
   *
   * @param textOut
   *          the text output device
   * @param textCase
   *          the text case
   * @param pruningMode
   *          the pruning mode
   * @param isBinary
   *          is this classifier binary
   * @param addInfos
   *          can we add infos and potentially insert citations?
   * @return the next case
   */
  static final ETextCase _printDescription(final ITextOutput textOut,
      final ETextCase textCase, final int pruningMode,
      final boolean isBinary, final boolean addInfos) {
    ETextCase nextCase;
    if (pruningMode == _WekaJ48ClassifierTrainingJob.PRUNING_OFF) {
      nextCase = textCase.appendWord("unpruned ", textOut); //$NON-NLS-1$
    } else {
      nextCase = textCase;
    }

    textOut.append("C4.5 "); //$NON-NLS-1$
    nextCase = nextCase.appendWord("classifier", textOut); //$NON-NLS-1$
    if (addInfos && (textOut instanceof IComplexText)) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, ETextCase.IN_SENTENCE,
              ESequenceMode.COMMA)) {
        builder.add(_WekaJ48Classifier.C45);
      }
    }

    if (addInfos) {
      textOut.append(" (Weka's J48 "); //$NON-NLS-1$
      nextCase = nextCase.appendWord("implementation", textOut);//$NON-NLS-1$
      if (textOut instanceof IComplexText) {
        try (final BibliographyBuilder builder = ((IComplexText) textOut)
            .cite(ECitationMode.ID, ETextCase.IN_SENTENCE,
                ESequenceMode.COMMA)) {
          builder.add(_WekaClassifier.WEKA);
        }
      }
      textOut.append(',');
      textOut.append(' ');
      nextCase = nextCase.appendWord("version", textOut);//$NON-NLS-1$
      textOut.appendNonBreakingSpace();
      textOut.append("3.8)");//$NON-NLS-1$
    }

    if (pruningMode == _WekaJ48ClassifierTrainingJob.PRUNING_OFF) {
      textOut.append(" with binary splits");//$NON-NLS-1$
    } else {
      textOut.append(" with ");//$NON-NLS-1$
      if (pruningMode == _WekaJ48ClassifierTrainingJob.PRUNING_REDUCED_ERROR) {
        textOut.append(" reduced-error");//$NON-NLS-1$
      }
      textOut.append(" pruning");//$NON-NLS-1$
      if (isBinary) {
        textOut.append(" and binary splits");//$NON-NLS-1$
      }
    }

    return nextCase;
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    return this.__printDescription(textOut, textCase, false);
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    textOut.append("J48");//$NON-NLS-1$
    return textCase.nextCase();
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    return this.__printDescription(textOut, textCase, true);
  }

  /** {@inheritDoc} */
  @Override
  public final String getPathComponentSuggestion() {
    String retVal;

    retVal = "j48"; //$NON-NLS-1$
    if (!(this.m_classifier.getUnpruned())) {
      if (this.m_classifier.getReducedErrorPruning()) {
        retVal += "Red";//$NON-NLS-1$
      }
      retVal += "Pruned";//$NON-NLS-1$
    }
    if (this.m_classifier.getBinarySplits()) {
      return retVal + "Bin";//$NON-NLS-1$
    }
    return retVal;
  }
}
