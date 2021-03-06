package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.nio.file.Path;
import java.util.Arrays;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.QuantileAggregate;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.StandardDeviationAggregate;

/** the outcome of multiple fitting applications */
public final class FittingOutcome implements Comparable<FittingOutcome> {

  /** the example data set */
  public final MultiFittingExampleDataset example;

  /** the single fitting outcomes */
  public final ArrayListView<SingleFittingOutcome> outcomes;

  /** the path where the log files are */
  public final Path path;

  /** the minimal value of each parameter */
  public final double[] min;
  /** the median value of each parameter */
  public final double[] median;
  /** the maximal value of each parameter */
  public final double[] max;
  /** the standard deviation of each parameter */
  public final double[] stddev;

  /** the minima of all errors */
  public final Errors minErrors;
  /** the medians of all errors */
  public final Errors medianErrors;
  /** the maxima of all errors */
  public final Errors maxErrors;
  /** the standard deviation of all errors */
  public final Errors stddevErrors;

  /**
   * create the fitting outcome
   *
   * @param _example
   *          the fitting example data set
   * @param results
   *          the results
   * @param _path
   *          the path
   * @param maxParams
   *          the maximum number of parameters
   */
  FittingOutcome(final MultiFittingExampleDataset _example,
      final int maxParams, final SingleFittingOutcome[] results,
      final Path _path) {
    super();

    int i;

    Arrays.sort(results);

    final StandardDeviationAggregate[] _stddev;
    final QuantileAggregate[] med;

    this.example = _example;
    this.path = _path;
    this.outcomes = new ArrayListView<>(results, false);

    this.min = new double[maxParams];
    this.max = new double[maxParams];
    this.median = new double[maxParams];
    this.stddev = new double[maxParams];

    i = Math.max(maxParams, 4);

    _stddev = new StandardDeviationAggregate[i];
    med = new QuantileAggregate[i];
    for (; (--i) >= 0;) {
      _stddev[i] = new StandardDeviationAggregate();
      med[i] = new QuantileAggregate(0.5d);
    }

    for (final SingleFittingOutcome o : results) {
      for (i = o.result.length; (--i) >= 0;) {
        _stddev[i].append(o.result[i]);
        med[i].append(o.result[i]);
      }
    }

    for (i = this.min.length; (--i) >= 0;) {
      this.min[i] = _stddev[i].getMinimum().doubleValue();
      this.max[i] = _stddev[i].getMaximum().doubleValue();
      this.stddev[i] = _stddev[i].doubleValue();
      _stddev[i].reset();
      this.median[i] = med[i].doubleValue();
      med[i].reset();
    }

    for (final SingleFittingOutcome o : results) {
      _stddev[0].append(o.errors.quality);
      med[0].append(o.errors.quality);
      _stddev[1].append(o.errors.rootMeanSquareError);
      med[1].append(o.errors.rootMeanSquareError);
      _stddev[2].append(o.errors.medianError);
      med[2].append(o.errors.medianError);
      _stddev[3].append(o.errors.runtime);
      med[3].append(o.errors.runtime);
    }

    this.minErrors = new Errors(_stddev[0].getMinimum().doubleValue(),
        _stddev[1].getMinimum().doubleValue(),
        _stddev[2].getMinimum().doubleValue(),
        _stddev[3].getMinimum().doubleValue());
    this.maxErrors = new Errors(_stddev[0].getMaximum().doubleValue(),
        _stddev[1].getMaximum().doubleValue(),
        _stddev[2].getMaximum().doubleValue(),
        _stddev[3].getMaximum().doubleValue());
    this.stddevErrors = new Errors(_stddev[0].doubleValue(),
        _stddev[1].doubleValue(), _stddev[2].doubleValue(),
        _stddev[3].doubleValue());
    this.medianErrors = new Errors(med[0].doubleValue(),
        med[1].doubleValue(), med[2].doubleValue(), med[3].doubleValue());
  }

  /**
   * Get all the result qualities as array.
   *
   * @return the quality array
   */
  final double[] _getQualities() {
    final double[] qualities;
    int index;

    index = this.outcomes.size();
    qualities = new double[index];
    for (; (--index) >= 0;) {
      qualities[index] = this.outcomes.get(index).errors.quality;
    }
    return qualities;
  }

  /**
   * Get all the result runtimes as array.
   *
   * @return the runtime array
   */
  final double[] _getRuntimes() {
    final double[] runtime;
    int index;

    index = this.outcomes.size();
    runtime = new double[index];
    for (; (--index) >= 0;) {
      runtime[index] = this.outcomes.get(index).errors.runtime;
    }
    return runtime;
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final FittingOutcome o) {
    int res;

    if (o == null) {
      return (-1);
    }
    if (o == this) {
      return 0;
    }

    res = this.medianErrors.compareTo(o.medianErrors);
    if (res != 0) {
      return res;
    }

    res = this.maxErrors.compareTo(o.maxErrors);
    if (res != 0) {
      return res;
    }

    return this.minErrors.compareTo(o.minErrors);
  }
}
