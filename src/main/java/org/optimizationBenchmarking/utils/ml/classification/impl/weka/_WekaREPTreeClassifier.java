package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.WekaTreeAccessor;

/**
 * The wrapper for Weka's {@link weka.classifiers.trees.REPTree}
 * classifiers classifier.
 */
final class _WekaREPTreeClassifier extends _WekaClassifier<REPTree> {

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
  _WekaREPTreeClassifier(final int[] selectedFeatures,
      final REPTree classifier, final _InternalInstance instance) {
    super(selectedFeatures, classifier, instance);
  }

  /** {@inheritDoc} */
  @Override
  public final void renderAsCode(
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    WekaTreeAccessor.renderREPTreeClassifier(this, this.m_selectedFeatures,
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
    return _WekaREPTreeClassifier._printDescription(textOut, textCase,
        this.m_classifier.getNoPruning(), addInfos);
  }

  /**
   * print the description
   *
   * @param textOut
   *          the text output device
   * @param textCase
   *          the text case
   * @param pruningOff
   *          is pruning turned off
   * @param addInfos
   *          can we add infos and potentially insert citations?
   * @return the next case
   */
  static final ETextCase _printDescription(final ITextOutput textOut,
      final ETextCase textCase, final boolean pruningOff,
      final boolean addInfos) {
    ETextCase nextCase;

    textOut.append("Weka's REPTree"); //$NON-NLS-1$
    nextCase = textCase.appendWord(" classifier", textOut); //$NON-NLS-1$

    if (addInfos) {
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

    if (pruningOff) {
      textOut.append(" without pruning");//$NON-NLS-1$
    } else {
      textOut.append(" with pruning");//$NON-NLS-1$
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
    textOut.append("REPTree");//$NON-NLS-1$
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

    retVal = "repTree"; //$NON-NLS-1$
    if (!(this.m_classifier.getNoPruning())) {
      retVal += "Pruned";//$NON-NLS-1$
    }
    return retVal;
  }
}
