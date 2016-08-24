package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import java.net.URI;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibBook;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibInCollection;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibTechReport;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.reflection.ReflectionUtils;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A curve fitter that uses {@link OptimizationBasedFittingJob}s. */
public abstract class OptimizationBasedFitter extends FunctionFitter {

  /** the first Levenberg-Marcquardt reference */
  private static final BibArticle REFERENCE_LEVENBERG_MARCQUARDT_1 = new BibArticle( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Kenneth", "Levenberg"), //$NON-NLS-1$//$NON-NLS-2$
  }), "A Method for the Solution of Certain Non-Linear Problems in Least Squares", //$NON-NLS-1$
      new BibDate(1944), "Quarterly of Applied Mathematics", //$NON-NLS-1$
      null, //
      "2", //$NON-NLS-1$
      null, //
      "164", //$NON-NLS-1$
      "168", //$NON-NLS-1$
      new BibOrganization(//
          "Brown University, Division of Applied Mathematics", //$NON-NLS-1$
          "Providence, RI, USA", null), //$NON-NLS-1$
      null, null);

  /** the second Levenberg-Marcquardt reference */
  private static final BibArticle REFERENCE_LEVENBERG_MARCQUARDT_2 = new BibArticle( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Donald W.", "Marquardt"), //$NON-NLS-1$//$NON-NLS-2$
  }), "An Algorithm for Least-Squares Estimation of Nonlinear Parameters", //$NON-NLS-1$
      new BibDate(1963, EBibMonth.JUNE),
      "SIAM Journal on Applied Mathematics", //$NON-NLS-1$
      "0036-1399", //$NON-NLS-1$
      "11", //$NON-NLS-1$
      "2", //$NON-NLS-1$
      "431", //$NON-NLS-1$
      "441", //$NON-NLS-1$
      new BibOrganization(//
          "Society for Industrial and Applied Mathematics (SIAM)", //$NON-NLS-1$
          "Philadelphia, PA, USA", null), //$NON-NLS-1$
      null, //
      "10.1137/0111030"); //$NON-NLS-1$

  /** the BOBYQA reference */
  private static final BibTechReport REFERENCE_BOBYQA = new BibTechReport( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Michael James David", "Powell"), //$NON-NLS-1$//$NON-NLS-2$
  }), "The BOBYQA Algorithm for Bound Constrained Optimization without Derivatives", //$NON-NLS-1$
      new BibDate(2009, EBibMonth.AUGUST), null, //
      "DAMTP2009/NA06", //$NON-NLS-1$
      null, //
      new BibOrganization(//
          "Department of Applied Mathematics and Theoretical Physics, Centre for Mathematical Sciences", //$NON-NLS-1$
          "Cambridge, England, UK", null), //$NON-NLS-1$
      URI.create(
          "http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf"), //$NON-NLS-1$
      null);

  /** the CMA-ES reference */
  private static final BibInCollection REFERENCE_CMAES = new BibInCollection( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Nikolaus", "Hansen"), //$NON-NLS-1$//$NON-NLS-2$
  }), "The CMA Evolution Strategy: A Comparing Review", //$NON-NLS-1$
      new BibBook( //
          BibAuthors.EMPTY_AUTHORS, //
          "Towards a New Evolutionary Computation \u2012 Advances in the Estimation of Distribution Algorithms", //$NON-NLS-1$
          new BibDate(2006), //
          new BibAuthors(new BibAuthor[] { //
              new BibAuthor("Jos\u00e9 Antonio", "Lozano"), //$NON-NLS-1$//$NON-NLS-2$
              new BibAuthor("Pedro", "Larra\u00f1aga"), //$NON-NLS-1$//$NON-NLS-2$
              new BibAuthor("I\u00f1aki", "Inza"), //$NON-NLS-1$//$NON-NLS-2$
              new BibAuthor("Endika", "Bengoetxea"), //$NON-NLS-1$//$NON-NLS-2$
  }), //
          new BibOrganization(//
              "Springer", //$NON-NLS-1$
              "Berlin Heidelberg", null), //$NON-NLS-1$
          "Studies in Fuzziness and Soft Computing", //$NON-NLS-1$
          "1434-9922", //$NON-NLS-1$
          "192", //$NON-NLS-1$
          null, //
          "978-3-540-29006-3", //$NON-NLS-1$
          null, //
          "10.1007/3-540-32494-1"), //$NON-NLS-1$
      "75", //$NON-NLS-1$
      "102", //$NON-NLS-1$
      "4", //$NON-NLS-1$
      null, //
      "10.1007/3-540-32494-1_4");//$NON-NLS-1$

  /** the Nelder-Mead reference */
  private static final BibArticle REFERENCE_NELDER_MEAD = new BibArticle( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("John Ashworth", "Nelder"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Roger", "Mead"), //$NON-NLS-1$//$NON-NLS-2$
  }), "A Simplex Method for Function Minimization", //$NON-NLS-1$
      new BibDate(1965), "The Computer Journal", //$NON-NLS-1$
      "1460-2067", //$NON-NLS-1$
      "7", //$NON-NLS-1$
      "4", //$NON-NLS-1$
      "308", //$NON-NLS-1$
      "313", //$NON-NLS-1$
      new BibOrganization(//
          "Oxford University Press", //$NON-NLS-1$
          "Oxford, UK", null), //$NON-NLS-1$
      null, //
      "10.1093/comjnl/7.4.308"); //$NON-NLS-1$

  /** the name of the Levenberg–Marquardt algorithm */
  protected static final String NAME_LEVENBERG_MARCQUARDT = "Levenberg-Marquardt"; //$NON-NLS-1$
  /** the name of the BOBYQA algorithm in short */
  protected static final String NAME_BOBYQA_SHORT = "BOBYQA"; //$NON-NLS-1$
  /** the name of the BOBYQA algorithm in lonf */
  protected static final String NAME_BOBYQA_LONG = "Bound Optimization BY Quadratic Approximation (BOBYQA)"; //$NON-NLS-1$
  /** the name of the CMAES algorithm in short */
  protected static final String NAME_CMAES_SHORT = "CMAES"; //$NON-NLS-1$
  /** the name of the CMAES algorithm in lonf */
  protected static final String NAME_CMAES_LONG = "Covariance Matrix Adaptation Evolution Strategy (CMAES)"; //$NON-NLS-1$
  /** the name of the Nelder-Mead algorithm */
  protected static final String NAME_NELDER_MEAD = "Nelder-Mead"; //$NON-NLS-1$

  /** the algorithm string */
  private static final String NAME_ALGORITHM = "algorithm";//$NON-NLS-1$

  /** the error */
  private final Throwable m_error;

  /** create */
  protected OptimizationBasedFitter() {
    super();

    Throwable cannot;

    cannot = null;
    try {
      ReflectionUtils.ensureClassesAreLoaded(//
          "org.apache.commons.math3.analysis.MultivariateFunction", //$NON-NLS-1$
          "org.apache.commons.math3.exception.MathIllegalStateException", //$NON-NLS-1$
          "org.apache.commons.math3.exception.NumberIsTooSmallException", //$NON-NLS-1$
          "org.apache.commons.math3.exception.OutOfRangeException", //$NON-NLS-1$
          "org.apache.commons.math3.exception.util.LocalizedFormats", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem", //$NON-NLS-1$
          "org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.linear.Array2DRowRealMatrix", //$NON-NLS-1$
          "org.apache.commons.math3.linear.ArrayRealVector", //$NON-NLS-1$
          "org.apache.commons.math3.linear.QRDecomposition", //$NON-NLS-1$
          "org.apache.commons.math3.linear.RealMatrix", //$NON-NLS-1$
          "org.apache.commons.math3.linear.RealVector", //$NON-NLS-1$
          "org.apache.commons.math3.optim.ConvergenceChecker", //$NON-NLS-1$
          "org.apache.commons.math3.optim.InitialGuess", //$NON-NLS-1$
          "org.apache.commons.math3.optim.MaxEval", //$NON-NLS-1$
          "org.apache.commons.math3.optim.MaxIter", //$NON-NLS-1$
          "org.apache.commons.math3.optim.PointValuePair", //$NON-NLS-1$
          "org.apache.commons.math3.optim.SimpleBounds", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.GoalType", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex", //$NON-NLS-1$
          "org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer", //$NON-NLS-1$
          "org.apache.commons.math3.random.JDKRandomGenerator", //$NON-NLS-1$
          "org.apache.commons.math3.util.FastMath", //$NON-NLS-1$
          "org.apache.commons.math3.util.Incrementor" //$NON-NLS-1$
      );

    } catch (final Throwable error) {
      cannot = error;
    }

    this.m_error = cannot;
  }

  /**
   * Print the name and title of the Levenberg-Marcquardt algorithm
   *
   * @param textCase
   *          the text case
   * @param textOut
   *          the text output destination
   * @param useShortName
   *          use the short name?
   * @return the next case
   */
  protected static final ETextCase printLevenbergMarcquardt(
      final ETextCase textCase, final ITextOutput textOut,
      final boolean useShortName) {
    ETextCase next;

    textOut.append(OptimizationBasedFitter.NAME_LEVENBERG_MARCQUARDT);
    next = textCase.nextCase();
    if (!useShortName) {
      textOut.append(' ');
      next = next.appendWord(OptimizationBasedFitter.NAME_ALGORITHM,
          textOut);
    }

    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder
            .add(OptimizationBasedFitter.REFERENCE_LEVENBERG_MARCQUARDT_1);
        builder
            .add(OptimizationBasedFitter.REFERENCE_LEVENBERG_MARCQUARDT_2);
      }
    }

    return next.nextCase();
  }

  /**
   * Print the name and title of the BOBYQA algorithm
   *
   * @param textCase
   *          the text case
   * @param textOut
   *          the text output destination
   * @param useShortName
   *          use the short name?
   * @return the next case
   */
  protected static final ETextCase printBOBYQA(final ETextCase textCase,
      final ITextOutput textOut, final boolean useShortName) {
    textOut.append(useShortName ? OptimizationBasedFitter.NAME_BOBYQA_SHORT
        : OptimizationBasedFitter.NAME_BOBYQA_LONG);

    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, textCase, ESequenceMode.COMMA)) {
        builder.add(OptimizationBasedFitter.REFERENCE_BOBYQA);
      }
    }

    return textCase.nextCase();
  }

  /**
   * Print the name and title of the CMAES algorithm
   *
   * @param textCase
   *          the text case
   * @param textOut
   *          the text output destination
   * @param useShortName
   *          use the short name?
   * @return the next case
   */
  protected static final ETextCase printCMAES(final ETextCase textCase,
      final ITextOutput textOut, final boolean useShortName) {
    textOut.append(useShortName ? OptimizationBasedFitter.NAME_CMAES_SHORT
        : OptimizationBasedFitter.NAME_CMAES_LONG);

    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, textCase, ESequenceMode.COMMA)) {
        builder.add(OptimizationBasedFitter.REFERENCE_CMAES);
      }
    }

    return textCase.nextCase();
  }

  /**
   * Print the name and title of the Nelder-Mead algorithm
   *
   * @param textCase
   *          the text case
   * @param textOut
   *          the text output destination
   * @param useShortName
   *          use the short name?
   * @return the next case
   */
  protected static final ETextCase printNelderMead(
      final ETextCase textCase, final ITextOutput textOut,
      final boolean useShortName) {
    ETextCase next;

    textOut.append(OptimizationBasedFitter.NAME_NELDER_MEAD);
    if (!useShortName) {
      textOut.append(' ');
      next = textCase.nextCase()
          .appendWord(OptimizationBasedFitter.NAME_ALGORITHM, textOut);
    } else {
      next = textCase.nextCase();
    }

    if (textOut instanceof IComplexText) {
      try (final BibliographyBuilder builder = ((IComplexText) textOut)
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(OptimizationBasedFitter.REFERENCE_NELDER_MEAD);
      }
    }

    return next.nextCase();
  }

  /**
   * Print the name and title of the simplex algorithm
   *
   * @param textCase
   *          the text case
   * @param textOut
   *          the text output destination
   * @return the next case
   */
  protected static final ETextCase printSimplex(final ETextCase textCase,
      final ITextOutput textOut) {
    return OptimizationBasedFitter.printNelderMead(textCase, textOut,
        true);
  }

  /** {@inheritDoc} */
  @Override
  public boolean canUse() {
    return (this.m_error == null);
  }

  /** {@inheritDoc} */
  @Override
  public void checkCanUse() {
    if (this.m_error != null) {
      throw new UnsupportedOperationException(//
          this.toString()
              + " cannot be used since required classes are missing in the classpath.", //$NON-NLS-1$
          this.m_error);
    }
    super.checkCanUse();
  }
}
