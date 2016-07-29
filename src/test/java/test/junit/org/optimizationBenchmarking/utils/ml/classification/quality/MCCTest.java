package test.junit.org.optimizationBenchmarking.utils.ml.classification.quality;

import org.junit.Assert;
import org.junit.Test;
import org.optimizationBenchmarking.utils.ml.classification.impl.quality.MCC;

import shared.junit.TestBase;

/** A test for the MCC quality measure. */
public class MCCTest extends TestBase {

  /** create the test */
  public MCCTest() {
    super();
  }

  /** test a diagonal matrix with value 15 */
  @Test(timeout = 3600000)
  public void test_diag_15() {

    Assert.assertEquals(1d,
        MCC.computeMCC(new int[][] { //
            { 15, 0, 0, 0 }, //
            { 0, 15, 0, 0 }, //
            { 0, 0, 15, 0 }, //
            { 0, 0, 0, 15 },//
    }), 1e-15d);
  }

  /**
   * test a matrix all values equal 5, except on the diagonal which is 0
   */
  @Test(timeout = 3600000)
  public void test_all_5_empty_diag() {

    Assert.assertEquals(-1d / 3d,
        MCC.computeMCC(new int[][] { //
            { 0, 5, 5, 5 }, //
            { 5, 0, 5, 5 }, //
            { 5, 5, 0, 5 }, //
            { 5, 5, 5, 0 },//
    }), 1e-15d);
  }

  /** test with the second column filled with 15 and the rest 0 */
  @Test(timeout = 3600000)
  public void test_col_2_15_rest_0() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 0, 15, 0, 0 }, //
            { 0, 15, 0, 0 }, //
            { 0, 15, 0, 0 }, //
            { 0, 15, 0, 0 },//
    }), 1e-15d);
  }

  /** test a matrix with some large and some small elements */
  @Test(timeout = 3600000)
  public void test_col_1_5000_5000_1() {

    Assert.assertEquals(-0.999d,
        MCC.computeMCC(new int[][] { //
            { 0, 0, 0, 1 }, //
            { 0, 0, 5000, 0 }, //
            { 0, 5000, 0, 0 }, //
            { 1, 0, 0, 0 },//
    }), 5e-4d);
  }

  /** test a 3x3 matrix with all ones */
  @Test(timeout = 3600000)
  public void test_all_1_3x3() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1 }, //
            { 1, 1, 1 }, //
            { 1, 1, 1 },//
    }), 1e-15d);
  }

  /** test a 4x4 matrix with all ones */
  @Test(timeout = 3600000)
  public void test_all_1_4x4() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 },//
    }), 1e-15d);
  }

  /** test a 5x5 matrix with all ones */
  @Test(timeout = 3600000)
  public void test_all_1_5x5() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1 },//
    }), 1e-15d);
  }

  /** test a 6x6 matrix with all ones */
  @Test(timeout = 3600000)
  public void test_all_1_6x6() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1, 1 }, //
            { 1, 1, 1, 1, 1, 1 },//
    }), 1e-15d);
  }

  /** test a matrix with all ones but one 10 value */
  @Test(timeout = 3600000)
  public void test_all_1_but_one_10() {

    Assert.assertEquals(-0.088d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 10, 1, 1, 1 },//
    }), 5e-4d);
  }

  /** test a matrix with all ones but one 100 value */
  @Test(timeout = 3600000)
  public void test_all_1_but_one_100() {

    Assert.assertEquals(-0.154d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 100, 1, 1, 1 },//
    }), 5e-4d);
  }

  /** test a matrix with all ones but one 1000 value */
  @Test(timeout = 3600000)
  public void test_all_1_but_one_1000() {

    Assert.assertEquals(-0.165d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1000, 1, 1, 1 },//
    }), 5e-4d);
  }

  /** test a matrix with all ones but one 1000 value */
  @Test(timeout = 3600000)
  public void test_all_1_but_one_10000() {

    Assert.assertEquals(-0.167d,
        MCC.computeMCC(new int[][] { //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 1, 1, 1, 1 }, //
            { 10000, 1, 1, 1 },//
    }), 5e-4d);
  }

  /** test a matrix with perfect classification */
  @Test(timeout = 3600000)
  public void test_ok_6() {

    Assert.assertEquals(1d,
        MCC.computeMCC(new int[][] { //
            { 6, 0 }, //
            { 0, 6 }, //
    }), 1e-15d);
  }

  /** test a matrix with almost perfect classification */
  @Test(timeout = 3600000)
  public void test_almost_ok_5_1() {

    Assert.assertEquals(2d / 3d,
        MCC.computeMCC(new int[][] { //
            { 5, 1 }, //
            { 1, 5 }, //
    }), 1e-15d);
  }

  /** test a matrix with ok 4/2 classification */
  @Test(timeout = 3600000)
  public void test_right_4_2() {

    Assert.assertEquals(1d / 3d,
        MCC.computeMCC(new int[][] { //
            { 4, 2 }, //
            { 2, 4 }, //
    }), 1e-15d);
  }

  /** test a matrix with all 3s */
  @Test(timeout = 3600000)
  public void test_all_3() {

    Assert.assertEquals(0d,
        MCC.computeMCC(new int[][] { //
            { 3, 3 }, //
            { 3, 3 }, //
    }), 1e-15d);
  }

  /** test a matrix with false 4/2 classification */
  @Test(timeout = 3600000)
  public void test_wrong_2_4() {

    Assert.assertEquals(-1d / 3d,
        MCC.computeMCC(new int[][] { //
            { 2, 4 }, //
            { 4, 2 }, //
    }), 1e-15d);
  }

  /** test a matrix with false 5/1 classification */
  @Test(timeout = 3600000)
  public void test_wrong_1_5() {

    Assert.assertEquals(-2d / 3d,
        MCC.computeMCC(new int[][] { //
            { 1, 5 }, //
            { 5, 1 }, //
    }), 1e-15d);
  }

  /** test a matrix with fully wrong 6 classification */
  @Test(timeout = 3600000)
  public void test_wrong_6() {

    Assert.assertEquals(-1d,
        MCC.computeMCC(new int[][] { //
            { 0, 6 }, //
            { 6, 0 }, //
    }), 1e-15d);
  }
}
