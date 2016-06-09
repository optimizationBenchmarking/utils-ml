package test.junit.org.optimizationBenchmarking.utils.ml.fitting;

import org.optimizationBenchmarking.utils.ml.fitting.impl.dels.DELSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;

import shared.junit.org.optimizationBenchmarking.utils.ml.fitting.ExampleFitterTest;

/** test the DE + least-squares */
public class DELSFitterTest extends ExampleFitterTest {

  /** create */
  public DELSFitterTest() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected IFunctionFitter getTool() {
    return DELSFitter.getInstance();
  }
}
