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
  /** the selected attributes */
  protected int[] m_selectedAttributes;

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
    int index, attributeSize;
    int[] attributes;
    int intValue;
    boolean booleanValue;
    double doubleValue;

    attributes = new int[this.m_featureTypes.length];
    index = (-1);
    attributeSize = 0;
    outer: for (final EFeatureType type : this.m_featureTypes) {
      ++index;

      switcher: switch (type) {
        case BOOLEAN: {
          booleanValue = ClassificationTools.featureDoubleToBoolean(
              this.m_knownSamples[0].featureValues[index]);
          for (final ClassifiedSample sample : this.m_knownSamples) {
            if (ClassificationTools.featureDoubleToBoolean(
                sample.featureValues[index]) != booleanValue) {
              break switcher;
            }
          }
          continue outer;
        }

        case NOMINAL: {
          intValue = ClassificationTools.featureDoubleToNominal(
              this.m_knownSamples[0].featureValues[index]);
          for (final ClassifiedSample sample : this.m_knownSamples) {
            if (ClassificationTools.featureDoubleToNominal(
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

      attributes[attributeSize++] = index;
    }

    if (attributeSize < attributes.length) {
      if (attributeSize <= 0) {
        return new _AllTheSameClass(
            ClassificationTools.getMostFrequentClass(this.m_knownSamples));
      }
      this.m_selectedAttributes = new int[attributeSize];
      System.arraycopy(attributes, 0, this.m_selectedAttributes, 0,
          attributeSize);
    } else {
      this.m_selectedAttributes = attributes;
    }

    result = super._invokeDoCall();
    this.m_selectedAttributes = null;
    return result;
  }
}
