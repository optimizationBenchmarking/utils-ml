package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDataClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleBasedDataClustererTest;

/** The test for R-based data clustering based on example data. */
public class RBasedExampleDataClustererTest
    extends ExampleBasedDataClustererTest {

  /** create */
  public RBasedExampleDataClustererTest() {
    super(RBasedDataClusterer.getInstance());
  }
}
