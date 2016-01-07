package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleDistanceClustererTest;

/** The test for R-based distance clustering. */
public class RBasedDistanceClustererTest
    extends ExampleDistanceClustererTest {

  /** create */
  public RBasedDistanceClustererTest() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IDistanceClusterer getTool() {
    return RBasedDistanceClusterer.getInstance();
  }
}
