package test.junit.org.optimizationBenchmarking.utils.ml.fitting;

import org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls.CMAESLSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;

import shared.junit.org.optimizationBenchmarking.utils.ml.fitting.ExampleFitterTest;

/** test the CMA-ES + least-squares */
public class CMAESLSFitterTest extends ExampleFitterTest {

  /** create */
  public CMAESLSFitterTest() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected IFunctionFitter getTool() {
    return CMAESLSFitter.getInstance();
  }
}
