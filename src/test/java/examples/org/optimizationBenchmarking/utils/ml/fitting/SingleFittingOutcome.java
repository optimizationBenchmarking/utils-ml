package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.nio.file.Path;

import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;

/** The result of a fitting process. */
public final class SingleFittingOutcome
    implements Comparable<SingleFittingOutcome> {

  /** the result of the fitting process */
  public final double[] result;
  /** the associated file */
  public final Path file;
  /** the fitting quality */
  public final Errors errors;
  /** the fitted model */
  public final ParametricUnaryFunction model;

  /**
   * create the fitting outcome
   *
   * @param _result
   *          the result of the fitting process
   * @param _file
   *          the associated file
   * @param _errors
   *          the errors the weighted median error
   * @param _model
   *          the fitted model
   */
  SingleFittingOutcome(final double[] _result, final Path _file,
      final Errors _errors, final ParametricUnaryFunction _model) {
    super();
    this.result = _result;
    this.file = _file;
    this.errors = _errors;
    this.model = _model;
  }

  /** {@inheritDoc} */
  @Override
  public final int compareTo(final SingleFittingOutcome o) {
    if (o == null) {
      return (-1);
    }
    if (o == this) {
      return 0;
    }
    return this.errors.compareTo(o.errors);
  }
}
