package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.tools.impl.abstr.ToolJobBuilder;

/** The abstract base class for classifier training job */
public class ClassifierTrainingJobBuilder extends
    ToolJobBuilder<IClassifierTrainingJob, ClassifierTrainingJobBuilder>
    implements IClassifierTrainingJobBuilder {

  /** the trainer */
  private final ClassifierTrainer m_trainer;

  /** the classifier quality measure */
  IClassifierQualityMeasure<?> m_qualityMeasure;
  /** the feature types */
  EFeatureType[] m_featureTypes;
  /** the known samples */
  ClassifiedSample[] m_knownSamples;

  /**
   * Create the job builder
   *
   * @param trainer
   *          the trainer
   */
  protected ClassifierTrainingJobBuilder(final ClassifierTrainer trainer) {
    super();
    if (trainer == null) {
      throw new IllegalArgumentException(
          "Owning trainer must not be null."); //$NON-NLS-1$
    }
    this.m_trainer = trainer;
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifierTrainingJob create() {
    return this.m_trainer._create(this);
  }

  /**
   * Check a classifier quality measure
   *
   * @param qualityMeasure
   *          the quality measure
   */
  static final void _checkClassifierQualityMeasure(
      final IClassifierQualityMeasure<?> qualityMeasure) {
    if (qualityMeasure == null) {
      throw new IllegalArgumentException(
          "Classifier quality measure cannot be null."); //$NON-NLS-1$
    }
  }

  /** {@inheritDoc} */
  @Override
  public final ClassifierTrainingJobBuilder setQualityMeasure(
      final IClassifierQualityMeasure<?> qualityMeasure) {
    ClassifierTrainingJobBuilder
        ._checkClassifierQualityMeasure(qualityMeasure);
    this.m_qualityMeasure = qualityMeasure;
    return this;
  }

  /**
   * Check the feature types
   *
   * @param featureTypes
   *          the feature types
   */
  static final void _checkFeatureTypesNotNull(
      final EFeatureType[] featureTypes) {
    if (featureTypes == null) {
      throw new IllegalArgumentException(
          "Feature types array cannot be null."); //$NON-NLS-1$
    }
  }

  /** {@inheritDoc} */
  @Override
  public final ClassifierTrainingJobBuilder setFeatureTypes(
      final EFeatureType... featureTypes) {
    int index;
    ClassifierTrainingJobBuilder._checkFeatureTypesNotNull(featureTypes);
    if (featureTypes.length <= 0) {
      throw new IllegalArgumentException(
          "There must be at least one feature type."); //$NON-NLS-1$
    }
    for (index = 0; index < featureTypes.length; ++index) {
      if (featureTypes[index] == null) {
        throw new IllegalArgumentException("Feature type at index " //$NON-NLS-1$
            + index + " is null.");//$NON-NLS-1$
      }
    }
    this.m_featureTypes = featureTypes;
    return this;
  }

  /**
   * Check the known samples
   *
   * @param knownSamples
   *          the known samples
   */
  static final void _checkKnownSamplesNotNull(
      final ClassifiedSample[] knownSamples) {
    if (knownSamples == null) {
      throw new IllegalArgumentException(
          "Known samples array cannot be null."); //$NON-NLS-1$
    }
  }

  /** {@inheritDoc} */
  @Override
  public final ClassifierTrainingJobBuilder setTrainingSamples(
      final ClassifiedSample... knownSamples) {
    int index, featureIndex;
    ClassifiedSample sample;

    if (this.m_featureTypes == null) {
      throw new IllegalStateException(//
          "Must first set list of feature types before setting list of known samples.");//$NON-NLS-1$
    }

    ClassifierTrainingJobBuilder._checkKnownSamplesNotNull(knownSamples);

    if (knownSamples.length <= 0) {
      throw new IllegalArgumentException(
          "There must be at least one known sample."); //$NON-NLS-1$
    }
    for (index = 0; index < knownSamples.length; ++index) {
      sample = knownSamples[index];
      if (sample == null) {
        throw new IllegalArgumentException("Known sample at index " //$NON-NLS-1$
            + index + " is null.");//$NON-NLS-1$
      }
      featureIndex = (-1);
      for (final EFeatureType type : this.m_featureTypes) {
        type.checkFeatureValue(sample.featureValues[++featureIndex]);
      }
    }

    this.m_knownSamples = knownSamples;
    return this;
  }
}
