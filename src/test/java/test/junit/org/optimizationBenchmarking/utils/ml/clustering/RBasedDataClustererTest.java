package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDataClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleDataClustererTest;

/** The test for R-based data clustering. */
public class RBasedDataClustererTest extends ExampleDataClustererTest {

  /** create */
  public RBasedDataClustererTest() {
    super(RBasedDataClusterer.getInstance());
  }
}
