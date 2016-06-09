package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.StableSum;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

/** An example data set for the fitting of data. */
public class MultiFittingExampleDataset
    implements Comparable<MultiFittingExampleDataset> {

  /** the name of the example data set */
  public final String name;

  /** the data matrix */
  public final IMatrix data;

  /** the model to be fitted */
  public final ArrayListView<ParametricUnaryFunction> models;

  /**
   * create the fitting example
   *
   * @param _name
   *          the name of the example data set
   * @param _data
   *          the data matrix
   * @param _models
   *          the models to be fitted
   */
  public MultiFittingExampleDataset(final String _name,
      final IMatrix _data,
      final ArrayListView<ParametricUnaryFunction> _models) {
    super();

    this.data = _data;
    this.models = _models;
    this.name = _name;
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return this.name + ": " + //$NON-NLS-1$
        this.models.toString() + " on " + //$NON-NLS-1$
        this.data.m() + " points"; //$NON-NLS-1$
  }

  /**
   * get a plot
   *
   * @param model
   *          the model to plot
   * @param fitting
   *          the fitting
   * @return the plot
   */
  final double[][] _plot(final ParametricUnaryFunction model,
      final double[] fitting) {
    final int size;
    double[][] dataArray;
    int index;
    double x, y, computedY;

    size = this.data.m();
    dataArray = new double[size][4];
    for (index = 0; index < size; index++) {
      dataArray[index][0] = x = this.data.getDouble(index, 0);
      dataArray[index][1] = y = this.data.getDouble(index, 1);
      dataArray[index][2] = computedY = model.value(x, fitting);
      computedY = Math.abs(computedY - y);
      y = Math.abs(y);
      if (y > 0d) {
        computedY /= y;
      }
      dataArray[index][3] = computedY;
    }
    Arrays.sort(dataArray, new __DblComparator());
    return dataArray;
  }

  /**
   * Plot the data and the fitting to the given file
   *
   * @param dest
   *          the destination path
   * @param model
   *          the model to plot
   * @param fitting
   *          the fitting
   * @param quality
   *          the quality of the fitting
   * @throws IOException
   *           if i/o fails
   */
  public final void plot(final Path dest,
      final ParametricUnaryFunction model, final double[] fitting,
      final double quality) throws IOException {
    final StableSum sum;
    double[][] dataArray;

    dataArray = this._plot(model, fitting);

    try (final OutputStream os = PathUtils.openOutputStream(dest)) {
      try (final OutputStreamWriter osw = new OutputStreamWriter(os)) {
        try (final BufferedWriter bw = new BufferedWriter(osw)) {
          sum = new StableSum();
          for (final double[] point : dataArray) {
            bw.write(Double.toString(point[0]));
            bw.write('\t');
            bw.write(Double.toString(point[1]));
            bw.write('\t');
            bw.write(Double.toString(point[2]));
            bw.write('\t');
            bw.write(Double.toString(point[3]));
            bw.newLine();
            sum.append(point[3]);
          }
          dataArray = null;

          bw.write('#');
          bw.newLine();
          bw.write('#');
          bw.newLine();
          bw.write("# fitting:"); //$NON-NLS-1$
          for (final double d : fitting) {
            bw.write('\t');
            bw.write(Double.toString(d));
          }
          bw.newLine();
          bw.write("# quality:\t"); //$NON-NLS-1$
          bw.write(Double.toString(quality));
          bw.newLine();
          bw.write("# error sum:\t"); //$NON-NLS-1$
          bw.write(Double.toString(sum.doubleValue()));
        }
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public final int hashCode() {
    return this.name.hashCode();
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final MultiFittingExampleDataset o) {
    return ((o == this) ? 0 : (this.name.compareTo(o.name)));
  }

  /** the internal comparator */
  private static final class __DblComparator
      implements Comparator<double[]> {
    /** create */
    __DblComparator() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final int compare(final double[] o1, final double[] o2) {
      int res;

      res = Double.compare(o1[0], o2[0]);
      if (res != 0) {
        return res;
      }
      res = Double.compare(o1[1], o2[1]);
      if (res != 0) {
        return res;
      }
      res = Double.compare(o1[2], o2[2]);
      if (res != 0) {
        return res;
      }
      return Double.compare(o1[3], o2[3]);
    }
  }
}
