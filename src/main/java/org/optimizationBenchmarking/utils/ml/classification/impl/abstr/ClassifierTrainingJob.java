package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.text.ITextable;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.MemoryTextOutput;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/** The abstract base class for classifier training jobs */
public abstract class ClassifierTrainingJob extends ToolJob
    implements IClassifierTrainingJob, ITextable {

  /** the classifier quality measure */
  protected IClassifierQualityMeasure<?> m_qualityMeasure;
  /** the feature types */
  protected EFeatureType[] m_featureTypes;
  /** the known samples */
  protected ClassifiedSample[] m_knownSamples;

  /**
   * Create the classifier training job
   *
   * @param builder
   *          the builder
   */
  protected ClassifierTrainingJob(
      final ClassifierTrainingJobBuilder builder) {
    super(builder);

    ClassifierTrainingJobBuilder._checkFeatureTypesNotNull(
        this.m_featureTypes = builder.m_featureTypes);
    ClassifierTrainingJobBuilder._checkKnownSamplesNotNull(
        this.m_knownSamples = builder.m_knownSamples);
    ClassifierTrainingJobBuilder._checkClassifierQualityMeasure(
        this.m_qualityMeasure = builder.m_qualityMeasure);
  }

  /**
   * Do the actual work.
   *
   * @return the classification result
   */
  protected abstract IClassifierTrainingResult doCall();

  /**
   * a small intermediate package-private hook to invoke {@link #doCall()}.
   *
   * @return the result
   */
  IClassifierTrainingResult _invokeDoCall() {
    return this.doCall();
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifierTrainingResult call()
      throws IllegalArgumentException {
    final Logger logger;
    String use;
    IClassifierTrainingResult result;

    logger = this.getLogger();
    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      use = this.toString();
      logger.finer("Beginning to execute " + use + '.');//$NON-NLS-1$
    } else {
      use = null;
    }

    result = null;
    try {
      result = this._invokeDoCall();
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        if (use == null) {
          use = this.toString();
        }
        logger.finer("Finished executing " + use + //$NON-NLS-1$
            ", obtained a classifier of type "//$NON-NLS-1$
            + result.getClassifier().toString() + ", quality " + //$NON-NLS-1$
            result.getQuality() + ", and complexity " + //$NON-NLS-1$
            result.getComplexity() + '.');
      }
    } catch (final Throwable error) {
      if (use == null) {
        use = this.toString();
      }
      throw new IllegalArgumentException("Failed to execute " + use//$NON-NLS-1$
          + '.', error);
    } finally {
      this.m_knownSamples = null;
      this.m_featureTypes = null;
      this.m_qualityMeasure = null;
    }

    ClassificationTools.checkClassifierTrainingResult(result);

    return result;
  }

  /**
   * Get the job name
   *
   * @return the job name
   */
  protected abstract String getJobName();

  /** {@inheritDoc} */
  @Override
  public void toText(final ITextOutput textOut) {
    final ClassifiedSample[] dataSamples;
    final EFeatureType[] featureTypes;
    final IClassifierQualityMeasure<?> qualityMeasure;

    textOut.append(this.getJobName());

    dataSamples = this.m_knownSamples;
    if ((dataSamples != null)) {
      textOut.append(" applied to a data set of ");//$NON-NLS-1$
      textOut.append(dataSamples.length);
      textOut.append(" samples");//$NON-NLS-1$
    }

    featureTypes = this.m_featureTypes;
    if (featureTypes != null) {
      textOut.append((dataSamples != null) ? " with " //$NON-NLS-1$
          : " to a data set with ");//$NON-NLS-1$
      textOut.append(featureTypes.length);
      textOut.append(" features");//$NON-NLS-1$
    }

    qualityMeasure = this.m_qualityMeasure;
    if (qualityMeasure != null) {
      textOut.append(" using quality measure ");//$NON-NLS-1$
      textOut.append(qualityMeasure);
    }
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    final MemoryTextOutput memText;

    memText = new MemoryTextOutput();
    this.toText(memText);
    return memText.toString();
  }
}
