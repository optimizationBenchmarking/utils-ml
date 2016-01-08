package test.junit.org.optimizationBenchmarking.utils.ml.fitting;

import org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex.LSSimplexFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;

import shared.junit.org.optimizationBenchmarking.utils.ml.fitting.ExampleFitterTest;

/** test the least-squares + simplex fitter */
public class LSSimplexFitterTest extends ExampleFitterTest {

  /** create */
  public LSSimplexFitterTest() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  protected IFunctionFitter getTool() {
    return LSSimplexFitter.getInstance();
  }
}
