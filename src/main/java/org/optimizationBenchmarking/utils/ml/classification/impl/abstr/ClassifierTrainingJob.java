package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/** The abstract base class for classifier training jobs */
public abstract class ClassifierTrainingJob extends ToolJob
    implements IClassifierTrainingJob {

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
   * get the textual representation of this object
   *
   * @return the textual representation
   */
  private final String __toText() {
    return " applying " + this.toString() + //$NON-NLS-1$
        " to a data set of " + this.m_knownSamples.length//$NON-NLS-1$
        + " samples with " + this.m_featureTypes.length//$NON-NLS-1$
        + " features and quality measure " + //$NON-NLS-1$
        this.m_qualityMeasure.toString();
  }

  /** {@iheritDoc} */
  @Override
  public final IClassifierTrainingResult call()
      throws IllegalArgumentException {
    final Logger logger;
    String use;
    IClassifierTrainingResult result;

    logger = this.getLogger();
    if ((logger != null) && (logger.isLoggable(Level.FINER))) {
      use = this.__toText();
      logger.finer("Beginning to" + use + '.');//$NON-NLS-1$
    } else {
      use = null;
    }

    result = null;
    try {
      result = this.doCall();
      if ((logger != null) && (logger.isLoggable(Level.FINER))) {
        if (use == null) {
          use = this.__toText();
        }
        logger.finer("Finished" + use + //$NON-NLS-1$
            ", obtained a classifier of type "//$NON-NLS-1$
            + result.getClassifier().toString() + " and quality " + //$NON-NLS-1$
            result.getQuality() + '.');
      }
    } catch (final Throwable error) {
      if (use == null) {
        use = this.__toText();
      }
      throw new IllegalArgumentException("Failed to apply" + use//$NON-NLS-1$
          + '.', error);
    } finally {
      this.m_knownSamples = null;
      this.m_featureTypes = null;
      this.m_qualityMeasure = null;
    }

    ClassificationTools.checkClassifierTrainingResult(result);

    return result;
  }
}
