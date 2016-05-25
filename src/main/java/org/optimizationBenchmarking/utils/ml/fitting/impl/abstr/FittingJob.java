package org.optimizationBenchmarking.utils.ml.fitting.impl.abstr;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.fitting.impl.FittingUtils;
import org.optimizationBenchmarking.utils.ml.fitting.quality.FittingQualityMeasure;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingJob;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingQualityMeasure;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.textOutput.MemoryTextOutput;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/** The fitting job */
public class FittingJob extends ToolJob implements IFittingJob {

  /** a function to fit */
  protected final ParametricUnaryFunction m_function;

  /** the fitting result */
  protected final FittingResult m_result;

  /** the matrix */
  protected final IMatrix m_data;

  /** the fitting quality measure */
  protected final IFittingQualityMeasure m_measure;

  /**
   * create the fitting job
   *
   * @param builder
   *          the fitting job builder
   */
  protected FittingJob(final FittingJobBuilder builder) {
    super(builder);

    FittingJobBuilder.validateFunction(//
        this.m_function = builder.getFunction());
    FittingQualityMeasure.validateData(//
        this.m_data = builder.getPoints());
    FittingJobBuilder.validateMeasure(//
        this.m_measure = builder.getQualityMeasure());

    this.m_result = new FittingResult(this.m_function);
  }

  /** Perform the fitting */
  protected void fit() {
    //
  }

  /**
   * Compute the quality of a given fitting.
   *
   * @param params
   *          the fitting, i.e., the parameters of the function to be
   *          fitted
   * @return the fitting quality, i.e., the weighted RMS sqrt(
   *         ((error*weight)^2) / n )
   */
  protected final double evaluate(final double[] params) {
    final double res;
    res = this.m_measure.evaluate(this.m_function, params);
    this.register(res, params);
    return res;
  }

  /**
   * Register a solution
   *
   * @param quality
   *          the solution quality
   * @param params
   *          the parameters
   */
  public final void register(final double quality, final double[] params) {
    if (quality < this.m_result.quality) {
      System.arraycopy(params, 0, this.m_result.solution, 0,
          this.m_result.solution.length);
      this.m_result.quality = quality;
    }
  }

  /**
   * create the basic message body
   *
   * @return the message body
   */
  private final MemoryTextOutput __createMessageBody() {
    final MemoryTextOutput textOut;

    textOut = new MemoryTextOutput(512);
    textOut.append(" function "); //$NON-NLS-1$
    FittingUtils.renderFunctionToFit(this.m_function, textOut);
    textOut.append(" on ");//$NON-NLS-1$
    textOut.append(this.m_data.m());
    textOut.append(" data points with method ");//$NON-NLS-1$
    textOut.append(this);
    textOut.append(" using quality measure ");//$NON-NLS-1$
    textOut.append(this.m_measure.toString());
    textOut.append(" with method ");//$NON-NLS-1$
    textOut.append(this.toString());
    return textOut;
  }

  /** {@inheritDoc} */
  @SuppressWarnings("null")
  @Override
  public final FittingResult call() throws IllegalArgumentException {
    final Logger logger;
    MemoryTextOutput textOut;
    Throwable error;
    String message;
    boolean canLog, isFinite;

    logger = this.getLogger();

    textOut = null;
    message = null;
    error = null;

    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      textOut = this.__createMessageBody();
      message = textOut.toString();
      logger.finer("Beginning to fit" + message);//$NON-NLS-1$
    }

    try {
      this.fit();

      canLog = (logger != null) && (logger.isLoggable(Level.FINER));
      isFinite = MathUtils.isFinite(this.m_result.quality);
      if (canLog || (!isFinite)) {
        if (textOut == null) {
          textOut = this.__createMessageBody();
        }
        textOut.append(", obtained result ");//$NON-NLS-1$
        FittingUtils.renderFittingResult(this.m_result.solution,
            this.m_result.quality, textOut);
        message = null;
      }

      if (isFinite) {
        if (canLog) {
          textOut.append('.');
          logger.finer("Finished fitting" + //$NON-NLS-1$
              textOut.toString());
        }
        return this.m_result;
      }
    } catch (final Throwable cause) {
      error = cause;
    }

    if (textOut == null) {
      textOut = this.__createMessageBody();
    }
    if (message == null) {
      message = textOut.toString();
    }
    message = ("Error while trying to fit" + message + '.'); //$NON-NLS-1$
    if (error != null) {
      throw new IllegalArgumentException(message, error);
    }
    throw new IllegalArgumentException(message);
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return this.getClass().getSimpleName();
  }
}
