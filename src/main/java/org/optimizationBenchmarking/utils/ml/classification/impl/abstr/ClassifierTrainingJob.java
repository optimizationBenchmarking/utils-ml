package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJob;

/** The abstract base class for classifier training jobs */
public class ClassifierTrainingJob extends ToolJob
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

  @Override
  public IClassifierTrainingResult call() throws IllegalArgumentException {
    // TODO Auto-generated method stub
    return null;
  }

}
