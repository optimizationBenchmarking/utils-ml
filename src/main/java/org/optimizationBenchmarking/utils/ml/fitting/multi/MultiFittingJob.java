package org.optimizationBenchmarking.utils.ml.fitting.multi;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.ml.fitting.impl.FittingUtils;
import org.optimizationBenchmarking.utils.ml.fitting.impl.abstr.FittingJobBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.quality.FittingQualityMeasure;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingQualityMeasure;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFittingResult;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.parallel.Execute;
import org.optimizationBenchmarking.utils.text.textOutput.MemoryTextOutput;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/**
 * A fitting job which tries to fit multiple models to a given data set and
 * picks the best result.
 */
public final class MultiFittingJob extends ToolJob
    implements Callable<IFittingResult> {

  /** the fitters */
  private final Iterable<IFunctionFitter> m_fitters;

  /** the {@code x-y}-coordinate pairs */
  private final IMatrix m_points;

  /** a functions to fit */
  private final Iterable<ParametricUnaryFunction> m_functions;

  /** the fitting quality measure */
  private final IFittingQualityMeasure m_measure;

  /**
   * create
   *
   * @param builder
   *          the owning builder
   */
  MultiFittingJob(final MultiFittingJobBuilder builder) {
    super(builder);

    FittingQualityMeasure.validateData(//
        this.m_points = builder.m_points);
    MultiFittingJobBuilder._validateFunctions(//
        this.m_functions = builder.m_functions);
    MultiFittingJobBuilder._validateFitters(//
        this.m_fitters = builder.m_fitters);
    FittingJobBuilder.validateMeasure(//
        this.m_measure = builder.m_measure);
  }

  /**
   * create the basic message body
   *
   * @return the message body
   */
  private final MemoryTextOutput __createMessageBody() {
    final MemoryTextOutput textOut;
    boolean first;

    textOut = new MemoryTextOutput(512);
    textOut.append(" function(s)"); //$NON-NLS-1$
    first = true;
    for (final ParametricUnaryFunction function : this.m_functions) {
      if (first) {
        first = false;
      } else {
        textOut.append(',');
      }
      textOut.append(' ');
      FittingUtils.renderFunctionToFit(function, textOut);
    }
    textOut.append(" on ");//$NON-NLS-1$
    textOut.append(this.m_points.m());
    textOut.append(" data points with method ");//$NON-NLS-1$
    textOut.append(this);
    textOut.append(" using quality measure ");//$NON-NLS-1$
    textOut.append(this.m_measure.toString());
    textOut.append(" with ");//$NON-NLS-1$
    textOut.append(this.toString());
    textOut.append(" using fitter(s)");//$NON-NLS-1$
    first = true;
    for (final IFunctionFitter fitter : this.m_fitters) {
      if (first) {
        first = false;
      } else {
        textOut.append(',');
      }
      textOut.append(' ');
      fitter.toText(textOut);
    }
    return textOut;
  }

  /** {@inheritDoc} */
  @Override
  public final IFittingResult call() {
    final ArrayList<Future<IFittingResult>> futures;
    final Logger logger;
    final int size;
    IFittingResult best, current;
    double bestQuality, curQuality;
    int bestLength, curLength;
    MemoryTextOutput textOut;
    Throwable error;
    String text;

    futures = new ArrayList<>();
    logger = this.getLogger();

    textOut = null;
    if ((logger != null) && (logger.isLoggable(Level.FINE))) {
      textOut = this.__createMessageBody();
      logger.finer("Beginning to fit" + textOut.toString());//$NON-NLS-1$
    }

    for (final IFunctionFitter fitter : this.m_fitters) {
      for (final ParametricUnaryFunction function : this.m_functions) {
        FittingJobBuilder.validateFunction(function);
        futures.add(Execute.parallel(fitter.use().setLogger(logger)//
            .setFunctionToFit(function)//
            .setQualityMeasure(this.m_measure)//
            .setPoints(this.m_points)//
            .create()));
      }
    }

    size = futures.size();
    if (size <= 0) {
      if (textOut == null) {
        textOut = this.__createMessageBody();
      }
      textOut.append(//
          "There must be at least one function fitter and at least one function to fit.");//$NON-NLS-1$
      throw new IllegalArgumentException(//
          "Error when fitting " + textOut.toString()); //$NON-NLS-1$
    }

    best = current = null;
    bestQuality = Double.POSITIVE_INFINITY;
    bestLength = Integer.MAX_VALUE;
    error = null;
    for (final Future<IFittingResult> future : futures) {

      try {
        current = future.get();
      } catch (final Throwable throwable) {
        if (best == null) {
          if (error == null) {
            error = throwable;
          } else {
            error.addSuppressed(throwable);
          }
        }
        continue;
      }

      if (current == null) {
        continue;
      }
      curQuality = current.getQuality();
      curLength = current.getFittedParametersRef().length;
      if ((best == null) || //
          ((curQuality >= 0d) && (curQuality < Double.POSITIVE_INFINITY)
              && (//
              (curQuality < bestQuality) || //
                  ((curQuality == bestQuality)
                      && (curLength < bestLength))))) {
        best = current;
        error = null;
        bestQuality = curQuality;
        bestLength = curLength;
      }
    }

    if ((best != null) && (bestQuality >= 0d)
        && (bestQuality < Double.POSITIVE_INFINITY)) {
      if ((logger != null) && (logger.isLoggable(Level.FINE))) {
        if (textOut == null) {
          textOut = this.__createMessageBody();
        }
        textOut.append(": best obtained fitting is ");//$NON-NLS-1$
        FittingUtils.renderFittingResult(best.getFittedParametersRef(),
            bestQuality, textOut);
        textOut.append(" for function ");//$NON-NLS-1$
        FittingUtils.renderFunctionToFit(best.getFittedFunction(),
            textOut);
      }
      return best;
    }

    if (textOut == null) {
      textOut = this.__createMessageBody();
    }
    textOut.append(//
        "No function could successfully be fitted to the data.");//$NON-NLS-1$
    text = ("Error when fitting: " + textOut.toString()); //$NON-NLS-1$
    if (error != null) {
      throw new IllegalArgumentException(text, error);
    }
    throw new IllegalArgumentException(text);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return MultiFunctionFitter.METHOD;
  }
}
