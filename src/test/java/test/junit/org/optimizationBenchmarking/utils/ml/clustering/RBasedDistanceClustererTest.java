package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDistanceClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleDistanceClustererTest;

/** The test for R-based distance clustering. */
public class RBasedDistanceClustererTest
    extends ExampleDistanceClustererTest {

  /** create */
  public RBasedDistanceClustererTest() {
    super(RBasedDistanceClusterer.getInstance());
  }
}
