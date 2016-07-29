package org.optimizationBenchmarking.utils.ml.classification.impl.quality;

import org.optimizationBenchmarking.utils.math.functions.arithmetic.Div;
import org.optimizationBenchmarking.utils.math.functions.power.Sqrt;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ConfusionMatrixBasedMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;

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

  /** create */
  private MCC() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final double evaluate(final IClassifier classifier,
      final int[][] token, final ClassifiedSample[] trainingSamples) {
    ConfusionMatrixBasedMeasure.fillInConfusionMatrix(classifier,
        trainingSamples, token);
    return (1d - (0.5d * MCC.computeMCC(token)));
  }

  /**
   * Compute the MCC based on a confusion matrix. This function returns the
   * un-normalized MCC, i.e., -1 means worst possible classification, 1 is
   * best. This is thus different from the result of
   * {@link #evaluate(IClassifier, int[][], ClassifiedSample[])}, which
   * returns a shifted and normalized value.
   *
   * @param C
   *          the confusion matrix
   * @return the MCC measure
   */
  public static final double computeMCC(final int[][] C) {
    int k, l, m, sumBelow11, sumBelow21;
    long sumAbove, totalSum, Ckk, Ckl, sumBelow1, sumBelow2;
    double sqrt1, sqrt2;

    sumAbove = sumBelow1 = sumBelow2 = 0L;

    totalSum = 0;
    for (final int[] row : C) {
      for (final int cell : row) {
        totalSum += cell;
      }
    }

    for (k = C.length; (--k) >= 0;) {
      Ckk = C[k][k];

      sumBelow11 = sumBelow21 = 0;

      for (l = C.length; (--l) >= 0;) {
        Ckl = C[k][l];
        for (m = C.length; (--m) >= 0;) {
          sumAbove += ((Ckk * C[l][m]) - (Ckl * C[m][k]));
        }

        sumBelow11 += Ckl;
        sumBelow21 += C[l][k];
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
