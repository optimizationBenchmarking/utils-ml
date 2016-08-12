package org.optimizationBenchmarking.utils.ml.fitting.impl.dels;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * This curve fitter uses a combination of least-squares solvers,
 * Differential Evolution, and Simplex Search to obtain high-quality
 * solutions.
 */
public final class DELSFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "Differential Evolution + Least-Squares + Simplex Fitter"; //$NON-NLS-1$

  /** the differential evolution reference */
  private static final BibArticle REFERENCE_DIFFERENTIAL_EVOLUTION = new BibArticle( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Rainer", "Storn"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Kenneth", "Price"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Differential Evolution \u2012 A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces", //$NON-NLS-1$
      new BibDate(1997, EBibMonth.DECEMBER),
      "Journal of Global Optimization", //$NON-NLS-1$
      "0925-5001", //$NON-NLS-1$
      "11", //$NON-NLS-1$
      "4", //$NON-NLS-1$
      "341", //$NON-NLS-1$
      "359", //$NON-NLS-1$
      new BibOrganization(//
          "Springer International Publishing AG", //$NON-NLS-1$
          "Cham, Switzerland", null), //$NON-NLS-1$
      null, "10.1023/A:1008202821328"); //$NON-NLS-1$

  /** create */
  DELSFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _DELSFittingJob(builder);
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;

    textOut.append("Differential Evolution"); //$NON-NLS-1$
    next = textCase.nextCase();
    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(DELSFitter.REFERENCE_DIFFERENTIAL_EVOLUTION);
      }
    }
    textOut.append('-');
    next = OptimizationBasedFitter.printLevenbergMarcquardt(next, textOut,
        true);
    textOut.append('/');
    next = OptimizationBasedFitter.printGaussNewton(next, textOut, true);
    textOut.append('-');
    next = OptimizationBasedFitter.printNelderMead(next, textOut, true);
    textOut.append(' ');
    return next.appendWord("hybrid", textOut); //$NON-NLS-1$
  }

  /**
   * Get the globally shared instance of the DE/LS-based curve fitter
   *
   * @return the instance of the DE/LS-based curve fitter
   */
  public static final DELSFitter getInstance() {
    return __DECurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return DELSFitter.METHOD;
  }

  /** the instance holder */
  private static final class __DECurveFitterHolder {
    /** the shared instance */
    static final DELSFitter INSTANCE = new DELSFitter();
  }
}
