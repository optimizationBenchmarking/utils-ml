package test.junit.org.optimizationBenchmarking.utils.ml.clustering;

import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDataClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusterer;

import shared.junit.org.optimizationBenchmarking.utils.ml.clustering.ExampleDataClustererTest;

/** The test for R-based data clustering. */
public class RBasedDataClustererTest extends ExampleDataClustererTest {

  /** create */
  public RBasedDataClustererTest() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected final IDataClusterer getTool() {
    return RBasedDataClusterer.getInstance();
  }
}
