package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.comparison.Compare;
import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.hash.HashUtils;
import org.optimizationBenchmarking.utils.text.Textable;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** A candidate solution of the fitting process. */
public class FittingCandidateSolution extends Textable
    implements Comparable<FittingCandidateSolution> {

  /** the fitting solution */
  public final double[] solution;

  /** the solution quality */
  public double quality;

  /**
   * Create the fitting candidate solution
   *
   * @param parameterCount
   *          the number of parameters
   */
  public FittingCandidateSolution(final int parameterCount) {
    super();
    this.solution = new double[parameterCount];
    this.quality = Double.POSITIVE_INFINITY;
  }

  /** {@inheritDoc} */
  @Override
  public int hashCode() {
    return HashUtils.combineHashes(//
        Arrays.hashCode(this.solution), //
        HashUtils.hashCode(this.quality));
  }

  /** {@inheritDoc} */
  @Override
  public void toText(final ITextOutput textOut) {
    char x;

    textOut.append(this.quality);
    textOut.append(':');
    x = '[';
    for (final double d : this.solution) {
      textOut.append(x);
      textOut.append(d);
      x = ',';
    }
    textOut.append(']');
  }

  /** {@inheritDoc} */
  @Override
  public boolean equals(final Object o) {
    final FittingCandidateSolution s;
    if (o instanceof FittingCandidateSolution) {
      s = ((FittingCandidateSolution) o);
      return ((EComparison.EQUAL.compare(this.quality, s.quality))
          && (Arrays.equals(this.solution, s.solution)));
    }
    return false;
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final FittingCandidateSolution o) {
    final int l1, l2;
    int res, index;

    if (o == this) {
      return 0;
    }
    if (o == null) {
      return (-1);
    }
    res = Compare.compare(this.quality, o.quality);
    if (res != 0) {
      return res;
    }

    l1 = this.solution.length;
    l2 = o.solution.length;
    if (l1 < l2) {
      return -1;
    }
    if (l1 > l2) {
      return 1;
    }

    index = (-1);
    for (final double d : this.solution) {
      res = Compare.compare(d, o.solution[++index]);
      if (res != 0) {
        return res;
      }
      if (index >= l1) {
        break;
      }
    }

    return 0;
  }

  /**
   * Copy a combination of solution and solution quality into this record
   *
   * @param _solution
   *          the solution to copy
   * @param _quality
   *          the quality to copy
   */
  protected void assign(final double[] _solution, final double _quality) {
    System.arraycopy(_solution, 0, this.solution, 0, this.solution.length);
    this.quality = _quality;
  }
}
