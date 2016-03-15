package shared.junit.org.optimizationBenchmarking.utils.ml.clustering;

import java.util.Random;

import org.junit.Ignore;
import org.optimizationBenchmarking.utils.math.matrix.AbstractMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.impl.IntMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.impl.LongMatrix1D;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringJobBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusteringJobBuilder;

/** The test for data clustering based on random data. */
@Ignore
public abstract class RandomDataClustererTest
    extends ClustererTestBasedOnRandomData<IDataClusterer> {

  /**
   * create
   *
   * @param clusterer
   *          the clusterer
   */
  public RandomDataClustererTest(final IDataClusterer clusterer) {
    super(clusterer);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final IntMatrix1D _randomIntMatrix(final int m,
      final int n, final Random random) {
    final int[] data;
    int i;

    i = m * n;
    data = new int[i];
    for (; (--i) >= 0;) {
      data[i] = random.nextInt();
    }

    return new IntMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final LongMatrix1D _randomLongMatrix(final int m,
      final int n, final Random random) {
    final long[] data;
    int i;

    i = m * n;
    data = new long[i];
    for (; (--i) >= 0;) {
      data[i] = random.nextLong();
    }

    return new LongMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  private static final DoubleMatrix1D _randomDoubleMatrix(final int m,
      final int n, final Random random) {
    final double[] data;
    double dvalue;
    int i;

    i = m * n;
    data = new double[i];
    for (; (--i) >= 0;) {
      do {
        dvalue = Double.longBitsToDouble(random.nextLong());
      } while ((dvalue != dvalue) || (dvalue <= Double.NEGATIVE_INFINITY)
          || (dvalue >= Double.POSITIVE_INFINITY));
      data[i] = dvalue;
    }

    return new DoubleMatrix1D(data, m, n);
  }

  /**
   * create the matrix
   *
   * @param m
   *          the number of rows (feature vectors)
   * @param n
   *          the number of cols (features)
   * @param random
   *          the ranom number generator
   * @return the matrix
   */
  static final AbstractMatrix _randomMatrix(final int m, final int n,
      final Random random) {
    switch (random.nextInt(3)) {
      case 0: {
        return RandomDataClustererTest._randomIntMatrix(m, n, random);
      }
      case 1: {
        return RandomDataClustererTest._randomLongMatrix(m, n, random);
      }
      default: {
        return RandomDataClustererTest._randomDoubleMatrix(m, n, random);
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  protected final void setRandomData(final IClusteringJobBuilder builder,
      final int elementCount, final Random random) {
    ((IDataClusteringJobBuilder) builder).setData(RandomDataClustererTest
        ._randomMatrix(elementCount, (1 + random.nextInt(20)), random));
  }
}
