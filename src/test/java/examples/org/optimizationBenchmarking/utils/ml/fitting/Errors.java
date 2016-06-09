package examples.org.optimizationBenchmarking.utils.ml.fitting;

import org.optimizationBenchmarking.utils.comparison.Compare;

/** measured errors */
public final class Errors implements Comparable<Errors> {

  /** the fitting quality */
  public final double quality;
  /** the mean square error */
  public final double rootMeanSquareError;
  /** the median error */
  public final double medianError;
  /** the measured runtime */
  public final double runtime;

  /**
   * create the errors
   *
   * @param _quality
   *          the fitting quality
   * @param _rootMeanSquareError
   *          the mean square error
   * @param _medianError
   *          the median error
   * @param _runtime
   *          the consumed runtime
   */
  Errors(final double _quality, final double _rootMeanSquareError,
      final double _medianError, final double _runtime) {
    super();
    this.quality = _quality;
    this.rootMeanSquareError = _rootMeanSquareError;
    this.medianError = _medianError;
    this.runtime = _runtime;
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final Errors o) {
    int res;

    if (o == null) {
      return (-1);
    }
    if (o == this) {
      return 0;
    }

    res = Compare.compare(this.quality, o.quality);
    if (res != 0) {
      return res;
    }

    res = Compare.compare(this.runtime, o.runtime);
    if (res != 0) {
      return res;
    }

    res = Compare.compare(this.rootMeanSquareError, o.rootMeanSquareError);
    if (res != 0) {
      return res;
    }

    return Compare.compare(this.medianError, o.medianError);
  }
}
