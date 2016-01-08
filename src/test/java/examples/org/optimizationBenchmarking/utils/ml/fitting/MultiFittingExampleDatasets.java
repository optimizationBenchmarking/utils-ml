package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.util.ArrayList;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

import shared.junit.org.optimizationBenchmarking.utils.ml.fitting.FittingExampleDataset;

/** Some examples for data fitting. */
public final class MultiFittingExampleDatasets {

  /** the examples */
  public static final ArrayListView<MultiFittingExampleDataset> EXAMPLES;

  static {

    final ArrayList<MultiFittingExampleDataset> list;

    list = new ArrayList<>();

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.A_1FlipHC_uf020_01_FOL, //
        FittingExampleDatasets.A_1FlipHC_uf020_01_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.A_1FlipHC_uf020_01_TOL, //
        FittingExampleDatasets.A_1FlipHC_uf020_01_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.A_1FlipHC_uf020_01_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.B_mFlipHC_uf100_01_FOL, //
        FittingExampleDatasets.B_mFlipHC_uf100_01_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.B_mFlipHC_uf100_01_TOL, //
        FittingExampleDatasets.B_mFlipHC_uf100_01_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.B_mFlipHC_uf100_01_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOL, //
        FittingExampleDatasets.C_2FlipHCrs_uf250_01_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOL, //
        FittingExampleDatasets.C_2FlipHCrs_uf250_01_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.C_2FlipHCrs_uf250_01_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_01_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_01_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_01_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_01_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_01_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_02_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_02_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_02_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_02_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_02_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_03_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_03_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_03_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_03_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_03_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_04_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_04_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_04_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_04_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_04_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_05_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_05_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_05_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_05_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_05_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_06_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_06_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_06_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_06_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_06_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_07_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_07_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_07_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_07_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_07_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_08_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_08_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_08_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_08_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_08_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_09_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_09_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_09_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_09_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_09_FTQ));

    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_10_FOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_10_FOE));
    list.add(MultiFittingExampleDatasets.__make(//
        FittingExampleDatasets.D_2FlipHC_uf250_10_TOL, //
        FittingExampleDatasets.D_2FlipHC_uf250_10_TOE));
    list.add(MultiFittingExampleDatasets
        .__make(FittingExampleDatasets.D_2FlipHC_uf250_10_FTQ));

    EXAMPLES = ArrayListView.collectionToView(list);
  }

  /**
   * create the fitting data set
   *
   * @param sets
   *          the data sets to join
   * @return the new set
   */
  private static final MultiFittingExampleDataset __make(
      final FittingExampleDataset... sets) {
    final ArrayList<ParametricUnaryFunction> func;

    func = new ArrayList<>();
    for (final FittingExampleDataset set : sets) {
      func.add(set.function);
    }

    return new MultiFittingExampleDataset(sets[0].name, sets[0].data, //
        ArrayListView.collectionToView(func));
  }
}
