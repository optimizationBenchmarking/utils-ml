package org.optimizationBenchmarking.utils.ml.fitting.quality;

import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingQualityMeasure;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** The base class for fitting quality measures */
public abstract class FittingQualityMeasure
    implements IFittingQualityMeasure {

  /** create the fitting quality measure */
  protected FittingQualityMeasure() {
    super();
  }

  /**
   * Validate the data matrix
   *
   * @param points
   *          the matrix' points
   */
  public static final void validateData(final IMatrix points) {
    if (points == null) {
      throw new IllegalArgumentException("Cannot set points to null."); //$NON-NLS-1$
    }
    if (points.m() <= 0) {
      throw new IllegalArgumentException(
          "Cannot set empty set of points.");//$NON-NLS-1$
    }
    if (points.n() != 2) {
      throw new IllegalArgumentException(
          "Point matrix must have n=2, but has n="//$NON-NLS-1$
              + points.n());
    }
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return this.getClass().getSimpleName();
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWords(this.toString(), textOut);
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
