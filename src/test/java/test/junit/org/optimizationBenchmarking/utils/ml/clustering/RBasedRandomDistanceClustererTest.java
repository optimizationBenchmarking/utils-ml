package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDistanceClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.RandomDistanceClustererTest;

/** The test for R-based distance clustering based on random data. */
public class RBasedRandomDistanceClustererTest
    extends RandomDistanceClustererTest {

  /** create */
  public RBasedRandomDistanceClustererTest() {
    super(RBasedDistanceClusterer.getInstance());
  }
}
