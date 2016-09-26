package org.optimizationBenchmarking.utils.ml.classification.impl.quality;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.math.functions.arithmetic.Div;
import org.optimizationBenchmarking.utils.math.functions.power.Sqrt;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ConfusionMatrix;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ConfusionMatrixBasedMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * A shifted and normalized version of the multi-class MCC measure defined
 * in
 * <p>
 * J. Gorodkin,
 * <em>Comparing two K-category assignments by a K-category correlation coefficient,</em>
 * Computational Biology and Chemistry, 28(5–6):367–374, December 2004,
 * doi:10.1016/j.compbiolchem.2004.09.006
 * </p>
 * The results of this measure are normalized into [0,1]. 0 means best
 * possible classification, 1 means worst.
 */
public final class MCC extends ConfusionMatrixBasedMeasure {

  /** The globally shared instance of this class. */
  public static final MCC INSTANCE = new MCC();

  /** The article for citing multi-class MCC */
  private static final BibArticle MCC_REFERENCE_1 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Jan", "Gorodkin"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Comparing Two K-Category Assignments by a K-Category Correlation Coefficient", //$NON-NLS-1$
      new BibDate(2004, EBibMonth.DECEMBER),
      "Computational Biology and Chemistry", //$NON-NLS-1$
      "1476-9271", //$NON-NLS-1$
      "28", //$NON-NLS-1$
      "5-6", //$NON-NLS-1$
      "367", //$NON-NLS-1$
      "374", //$NON-NLS-1$
      new BibOrganization(//
          "Elsevier Science Publishers B.V.", //$NON-NLS-1$
          "Amsterdam, The Netherlands", null), //$NON-NLS-1$
      null, "10.1016/j.compbiolchem.2004.09.006");//$NON-NLS-1$

  /** The article for citing MCC */
  private static final BibArticle MCC_REFERENCE_2 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("B. W.", "Matthews"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme", //$NON-NLS-1$
      new BibDate(1995, EBibMonth.OCTOBER, 20),
      "Biochimica et Biophysica Acta (BBA) - Protein Structure", //$NON-NLS-1$
      null, //
      "405", //$NON-NLS-1$
      "2", //$NON-NLS-1$
      "442", //$NON-NLS-1$
      "451", //$NON-NLS-1$
      new BibOrganization(//
          "Elsevier Science Publishers B.V.", //$NON-NLS-1$
          "Amsterdam, The Netherlands", null), //$NON-NLS-1$
      null, "10.1016/0005-2795(75)90109-9");//$NON-NLS-1$

  /** create */
  private MCC() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final IClassifier classifier,
      final ConfusionMatrix token,
      final ClassifiedSample[] trainingSamples) {
    token.fillInConfusionMatrix(classifier, trainingSamples);
    return (1d - (0.5d * MCC.computeMCC(token)));
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWord("multi-class MCC", textOut); //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWord(this.toString(), textOut);
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;

    next = this.printLongName(textOut, textCase);

    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = (((IComplexText) textOut)
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA))) {
        builder.add(MCC.MCC_REFERENCE_1);
        builder.add(MCC.MCC_REFERENCE_2);
      }
    }

    return next;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "multi-class Matthews Correlation Coefficient (MCC)"; //$NON-NLS-1$
  }

  /**
   * Compute the MCC based on a confusion matrix. This function returns the
   * un-normalized MCC, i.e., -1 means worst possible classification, 1 is
   * best. This is thus different from the result of
   * {@link #evaluate(IClassifier, ConfusionMatrix, ClassifiedSample[])},
   * which returns a shifted and normalized value.
   *
   * @param C
   *          the confusion matrix
   * @return the MCC measure
   */
  public static final double computeMCC(final ConfusionMatrix C) {
    final int length;
    int k, l, m, sumBelow11, sumBelow21, totalSum;
    long sumAbove, Ckk, Ckl, sumBelow1, sumBelow2;
    double sqrt1, sqrt2;

    sumAbove = sumBelow1 = sumBelow2 = 0L;

    totalSum = C.getSampleSize();

    length = C.getClassCount();
    for (k = length; (--k) >= 0;) {
      Ckk = C.getConfusionForIndexClasses(k, k);

      sumBelow11 = sumBelow21 = 0;

      for (l = length; (--l) >= 0;) {
        Ckl = C.getConfusionForIndexClasses(k, l);
        for (m = length; (--m) >= 0;) {
          sumAbove += ((Ckk * C.getConfusionForIndexClasses(l, m))
              - (Ckl * C.getConfusionForIndexClasses(m, k)));
        }

        sumBelow11 += Ckl;
        sumBelow21 += C.getConfusionForIndexClasses(l, k);
      }

      sumBelow1 += (sumBelow11 * (totalSum - sumBelow11));
      sumBelow2 += (sumBelow21 * (totalSum - sumBelow21));
    }

    // There seems to be a weird special case where one column is filled
    // with zeros, in which case we get 0 below the fraction. However, for
    // this case, MCC is defined to be 0.
    if ((sumBelow1 == 0L) || (sumBelow2 == 0L)) {
      return 0d;
    }

    sqrt1 = Sqrt.INSTANCE.computeAsDouble(sumBelow1);
    sqrt2 = Sqrt.INSTANCE.computeAsDouble(sumBelow2);

    // try to compute at maximum precision
    sumBelow1 = ((long) sqrt1);
    sumBelow2 = ((long) sqrt2);
    if ((sumBelow1 == sqrt1) && (sumBelow2 == sqrt2)) {
      return Div.INSTANCE.computeAsDouble(sumAbove, sumBelow1 * sumBelow2);
    }
    return Div.INSTANCE.computeAsDouble(sumAbove, sqrt1 * sqrt2);
  }
}
