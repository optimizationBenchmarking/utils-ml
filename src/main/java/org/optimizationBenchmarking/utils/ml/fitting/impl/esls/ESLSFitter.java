package org.optimizationBenchmarking.utils.ml.fitting.impl.esls;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.OptimizationBasedFitter;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * This curve fitter uses a combination of least-squares solvers and a
 * (mu+lambda)-Evolution Strategy to obtain high-quality solutions.
 */
public final class ESLSFitter extends OptimizationBasedFitter {

  /** the method name */
  static final String METHOD = "ES + Least-Squares Fitter"; //$NON-NLS-1$

  /** the evolution strategy reference */
  private static final BibArticle REFERENCE_EVOLUTION_STRATEGY = new BibArticle( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Hans-Georg", "Beyer"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Hans-Paul", "Schwefel"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Evolution Strategies \u2012 A Comprehensive Introduction", //$NON-NLS-1$
      new BibDate(2002, EBibMonth.MARCH),
      "Natural Computing: An International Journal", //$NON-NLS-1$
      "1567-78181", //$NON-NLS-1$
      "1", //$NON-NLS-1$
      "1", //$NON-NLS-1$
      "3", //$NON-NLS-1$
      "52", //$NON-NLS-1$
      new BibOrganization(//
          "Springer Netherlands", //$NON-NLS-1$
          "Dordrecht, Netherlands", null), //$NON-NLS-1$
      null, "10.1023/A:1015059928466"); //$NON-NLS-1$

  /** create */
  ESLSFitter() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final FittingJob create(final FittingJobBuilder builder) {
    return new _ESLSFittingJob(builder);
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    ETextCase next;

    if (textOut instanceof IComplexText) {
      try (final IMath math = ((IComplexText) textOut).inlineMath()) {
        try (final IMath braces = math.inBraces()) {
          try (final IMath add = braces.add()) {
            try (final IText name = add.name()) {
              name.append('\u03bc');
            }
            try (final IText name = add.name()) {
              name.append('\u03bb');
            }
          }
        }
      }
    } else {
      textOut.append("(\u03bc+\u03bb)");//$NON-NLS-1$
    }
    textOut.appendNonBreakingSpace();

    textOut.append("Evolution Strategy"); //$NON-NLS-1$
    next = textCase.nextCase();
    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(ESLSFitter.REFERENCE_EVOLUTION_STRATEGY);
      }
    }
    textOut.append('-');
    next = OptimizationBasedFitter.printLevenbergMarcquardt(next, textOut,
        true);
    textOut.append('/');
    next = OptimizationBasedFitter.printGaussNewton(next, textOut, true);
    textOut.append(' ');
    return next.appendWord("hybrid", textOut); //$NON-NLS-1$
  }

  /**
   * Get the globally shared instance of the ES/LS-based curve fitter
   *
   * @return the instance of the ES/LS-based curve fitter
   */
  public static final ESLSFitter getInstance() {
    return __ESLSCurveFitterHolder.INSTANCE;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return ESLSFitter.METHOD;
  }

  /** the instance holder */
  private static final class __ESLSCurveFitterHolder {
    /** the shared instance */
    static final ESLSFitter INSTANCE = new ESLSFitter();
  }
}
