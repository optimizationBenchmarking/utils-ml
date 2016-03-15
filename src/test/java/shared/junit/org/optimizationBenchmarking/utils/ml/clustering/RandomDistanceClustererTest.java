package shared.junit.org.optimizationBenchmarking.utils.ml.clustering;

import java.util.Random;

import org.junit.Ignore;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DistanceMatrixBuilderJob;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.IAggregate;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.EuclideanDistance;
import org.optimizationBenchmarking.utils.ml.clustering.impl.dist.MeasureBasedDistanceMatrixBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IClusteringJobBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusteringJobBuilder;

/** The test for data clusterin based on random datag. */
@Ignore
public abstract class RandomDistanceClustererTest
    extends ClustererTestBasedOnRandomData<IDistanceClusterer> {

  /**
   * create
   *
   * @param clusterer
   *          the clusterer
   */
  public RandomDistanceClustererTest(final IDistanceClusterer clusterer) {
    super(clusterer);
  }

  /** {@inheritDoc} */
  @Override
  protected final void setRandomData(final IClusteringJobBuilder builder,
      final int elementCount, final Random random) {
    IMatrix distance;
    double dvalue;
    int i, j;

    looper: for (;;) {
      if (random.nextBoolean()) {
        distance = new __RandomDistanceMatrixBuilder(elementCount, random)
            .call();
      } else {
        distance = new MeasureBasedDistanceMatrixBuilder(//
            RandomDataClustererTest._randomMatrix(elementCount,
                (1 + random.nextInt(20)), random),
            new EuclideanDistance()).call();
      }

      for (i = elementCount; (--i) >= 0;) {
        for (j = elementCount; (--j) >= 0;) {
          dvalue = distance.getDouble(i, j);
          if ((dvalue != dvalue) || (dvalue <= Double.NEGATIVE_INFINITY)
              || (dvalue >= Double.POSITIVE_INFINITY)) {
            continue looper;
          }
        }
      }
      break looper;
    }

    ((IDistanceClusteringJobBuilder) builder).setDistanceMatrix(distance);
  }

  /** the random distance matrix builder */
  private static final class __RandomDistanceMatrixBuilder
      extends DistanceMatrixBuilderJob {
    /** the m */
    final int m_m;
    /** the random number generator */
    final Random m_random;
    /** the mode */
    final int m_mode;

    /**
     * create
     *
     * @param m
     *          the m
     * @param random
     *          the random
     */
    __RandomDistanceMatrixBuilder(final int m, final Random random) {
      super();
      this.m_m = m;
      this.m_mode = random.nextInt(3);
      this.m_random = random;
    }

    /** {@inheritDoc} */
    @Override
    protected final int getElementCount() {
      return this.m_m;
    }

    /** {@inheritDoc} */
    @Override
    protected final void setDistance(final int i, final int j,
        final IAggregate appendTo) {
      double dvalue;
      switch (this.m_mode) {
        case 0: {
          appendTo.append(this.m_random.nextInt());
          return;
        }
        case 1: {
          appendTo.append(this.m_random.nextLong());
          return;
        }
        default: {
          do {
            dvalue = Double.longBitsToDouble(this.m_random.nextLong());
          } while ((dvalue != dvalue)
              || (dvalue <= Double.NEGATIVE_INFINITY)
              || (dvalue >= Double.POSITIVE_INFINITY));
          appendTo.append(dvalue);
        }
      }
    }
  }
}
