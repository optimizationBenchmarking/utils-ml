package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDistanceClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleBasedDistanceClustererTest;

/** The test for R-based distance clustering based on example data. */
public class RBasedExampleDistanceClustererTest
    extends ExampleBasedDistanceClustererTest {

  /** create */
  public RBasedExampleDistanceClustererTest() {
    super(RBasedDistanceClusterer.getInstance());
  }
}
