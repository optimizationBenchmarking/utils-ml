package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDataClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.RandomDataClustererTest;

/** The test for R-based data clustering based on random data. */
public class RBasedRandomDataClustererTest
    extends RandomDataClustererTest {

  /** create */
  public RBasedRandomDataClustererTest() {
    super(RBasedDataClusterer.getInstance());
  }
}
