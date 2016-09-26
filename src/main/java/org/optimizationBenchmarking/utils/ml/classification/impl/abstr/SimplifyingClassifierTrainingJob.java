package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.comparison.Compare;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

/**
 * This is a classifying training job which first tries to simplify all the
 * data it receives in order to speed up classification.
 */
public abstract class SimplifyingClassifierTrainingJob
    extends ClassifierTrainingJob {
  /** the selected features */
  protected int[] m_selectedFeatures;

  /**
   * Create the classifier training job
   *
   * @param builder
   *          the builder
   */
  protected SimplifyingClassifierTrainingJob(
      final ClassifierTrainingJobBuilder builder) {
    super(builder);
  }

  /** {@inheritDoc} */
  @Override
  final IClassifierTrainingResult _invokeDoCall() {
    final IClassifierTrainingResult result;
    int index, featureSize;
    int[] features;
    int intValue;
    Boolean booleanValue;
    double doubleValue;

    features = new int[this.m_featureTypes.length];
    index = (-1);
    featureSize = 0;
    outer: for (final EFeatureType type : this.m_featureTypes) {
      ++index;

      switcher: switch (type) {
        case BOOLEAN: {
          booleanValue = EFeatureType.featureDoubleToBoolean(
              this.m_knownSamples[0].featureValues[index]);
          for (final ClassifiedSample sample : this.m_knownSamples) {
            if (EFeatureType.featureDoubleToBoolean(
                sample.featureValues[index]) != booleanValue) {
              break switcher;
            }
          }
          continue outer;
        }

        case NOMINAL: {
          intValue = EFeatureType.featureDoubleToNominal(
              this.m_knownSamples[0].featureValues[index]);
          for (final ClassifiedSample sample : this.m_knownSamples) {
            if (EFeatureType.featureDoubleToNominal(
                sample.featureValues[index]) != intValue) {
              break switcher;
            }
          }
          continue outer;
        }

        case NUMERICAL: {
          doubleValue = this.m_knownSamples[0].featureValues[index];
          for (final ClassifiedSample sample : this.m_knownSamples) {
            if (!(Compare.equals(sample.featureValues[index],
                doubleValue))) {
              break switcher;
            }
          }
          continue outer;
        }

        default: {
          throw new IllegalStateException("Cannot deal with " + type); //$NON-NLS-1$
        }
      }

      features[featureSize++] = index;
    }

    if ((featureSize < features.length) || (featureSize <= 0)) {
      if (featureSize <= 0) {
        return new _AllTheSameClass(
            new ClassifiedSampleInfo(this.m_knownSamples)
                .getBiggestSampleClass());
      }
      this.m_selectedFeatures = new int[featureSize];
      System.arraycopy(features, 0, this.m_selectedFeatures, 0,
          featureSize);
    } else {
      this.m_selectedFeatures = features;
    }

    result = super._invokeDoCall();
    this.m_selectedFeatures = null;
    return result;
  }
}
