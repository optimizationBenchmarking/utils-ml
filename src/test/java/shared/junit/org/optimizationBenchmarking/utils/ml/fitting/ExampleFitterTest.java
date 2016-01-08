package shared.junit.org.optimizationBenchmarking.utils.ml.fitting;

import org.junit.Ignore;
import org.junit.Test;
import org.optimizationBenchmarking.utils.ml.fitting.quality.WeightedRootMeanSquareError;

import examples.org.optimizationBenchmarking.utils.ml.fitting.FittingExampleDatasets;

/** A test for function fitters based on examples */
@Ignore
public abstract class ExampleFitterTest extends FitterTest {

  /** create the test */
  protected ExampleFitterTest() {
    super();
  }

  /** test A_1FlipHC_uf020_01_FOL */
  @Test(timeout = 3600000)
  public void test_A_1FlipHC_uf020_01_FOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.A_1FlipHC_uf020_01_FOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.A_1FlipHC_uf020_01_FOL.data));
  }

  /** test A_1FlipHC_uf020_01_FOE */
  @Test(timeout = 3600000)
  public void test_A_1FlipHC_uf020_01_FOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.A_1FlipHC_uf020_01_FOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.A_1FlipHC_uf020_01_FOE.data));
  }

  /** test A_1FlipHC_uf020_01_TOL */
  @Test(timeout = 3600000)
  public void test_A_1FlipHC_uf020_01_TOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.A_1FlipHC_uf020_01_TOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.A_1FlipHC_uf020_01_TOL.data));
  }

  /** test A_1FlipHC_uf020_01_TOE */
  @Test(timeout = 3600000)
  public void test_A_1FlipHC_uf020_01_TOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.A_1FlipHC_uf020_01_TOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.A_1FlipHC_uf020_01_TOE.data));
  }

  /** test A_1FlipHC_uf020_01_FTQ */
  @Test(timeout = 3600000)
  public void test_A_1FlipHC_uf020_01_FTQ_wrmsq() {
    this.fitExample(FittingExampleDatasets.A_1FlipHC_uf020_01_FTQ, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.A_1FlipHC_uf020_01_FTQ.data));
  }

  /** test B_mFlipHC_uf100_01_FOL */
  @Test(timeout = 3600000)
  public void test_B_mFlipHC_uf100_01_FOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.B_mFlipHC_uf100_01_FOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.B_mFlipHC_uf100_01_FOL.data));
  }

  /** test B_mFlipHC_uf100_01_FOE */
  @Test(timeout = 3600000)
  public void test_B_mFlipHC_uf100_01_FOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.B_mFlipHC_uf100_01_FOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.B_mFlipHC_uf100_01_FOE.data));
  }

  /** test B_mFlipHC_uf100_01_TOL */
  @Test(timeout = 3600000)
  public void test_B_mFlipHC_uf100_01_TOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.B_mFlipHC_uf100_01_TOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.B_mFlipHC_uf100_01_TOL.data));
  }

  /** test B_mFlipHC_uf100_01_TOE */
  @Test(timeout = 3600000)
  public void test_B_mFlipHC_uf100_01_TOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.B_mFlipHC_uf100_01_TOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.B_mFlipHC_uf100_01_TOE.data));
  }

  /** test B_mFlipHC_uf100_01_FTQ */
  @Test(timeout = 3600000)
  public void test_B_mFlipHC_uf100_01_FTQ_wrmsq() {
    this.fitExample(FittingExampleDatasets.B_mFlipHC_uf100_01_FTQ, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.B_mFlipHC_uf100_01_FTQ.data));
  }

  /** test C_2FlipHCrs_uf250_01_FOL */
  @Test(timeout = 3600000)
  public void test_C_2FlipHCrs_uf250_01_FOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOL.data));
  }

  /** test C_2FlipHCrs_uf250_01_FOE */
  @Test(timeout = 3600000)
  public void test_C_2FlipHCrs_uf250_01_FOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOE.data));
  }

  /** test C_2FlipHCrs_uf250_01_TOL */
  @Test(timeout = 3600000)
  public void test_C_2FlipHCrs_uf250_01_TOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOL.data));
  }

  /** test C_2FlipHCrs_uf250_01_TOE */
  @Test(timeout = 3600000)
  public void test_C_2FlipHCrs_uf250_01_TOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOE.data));
  }

  /** test C_2FlipHCrs_uf250_01_FTQ */
  @Test(timeout = 3600000)
  public void test_C_2FlipHCrs_uf250_01_FTQ_wrmsq() {
    this.fitExample(FittingExampleDatasets.C_2FlipHCrs_uf250_01_FTQ, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.C_2FlipHCrs_uf250_01_FTQ.data));
  }

  /** test D_2FlipHC_uf250_01_FOL */
  @Test(timeout = 3600000)
  public void test_D_2FlipHC_uf250_01_FOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.D_2FlipHC_uf250_01_FOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.D_2FlipHC_uf250_01_FOL.data));
  }

  /** test D_2FlipHC_uf250_01_FOE */
  @Test(timeout = 3600000)
  public void test_D_2FlipHC_uf250_01_FOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.D_2FlipHC_uf250_01_FOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.D_2FlipHC_uf250_01_FOE.data));
  }

  /** test D_2FlipHC_uf250_01_TOL */
  @Test(timeout = 3600000)
  public void test_D_2FlipHC_uf250_01_TOL_wrmsq() {
    this.fitExample(FittingExampleDatasets.D_2FlipHC_uf250_01_TOL, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.D_2FlipHC_uf250_01_TOL.data));
  }

  /** test D_2FlipHC_uf250_01_TOE */
  @Test(timeout = 3600000)
  public void test_D_2FlipHC_uf250_01_TOE_wrmsq() {
    this.fitExample(FittingExampleDatasets.D_2FlipHC_uf250_01_TOE, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.D_2FlipHC_uf250_01_TOE.data));
  }

  /** test D_2FlipHC_uf250_01_FTQ */
  @Test(timeout = 3600000)
  public void test_D_2FlipHC_uf250_01_FTQ_wrmsq() {
    this.fitExample(FittingExampleDatasets.D_2FlipHC_uf250_01_FTQ, //
        new WeightedRootMeanSquareError(//
            FittingExampleDatasets.D_2FlipHC_uf250_01_FTQ.data));
  }
}
