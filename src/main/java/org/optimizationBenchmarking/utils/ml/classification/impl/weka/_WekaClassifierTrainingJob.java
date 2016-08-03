package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import java.util.ArrayList;

import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingResult;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * a classifier training job wrapping around Weka
 *
 * @param <CT>
 *          the classifier type
 */
abstract class _WekaClassifierTrainingJob<CT extends Classifier>
    extends ClassifierTrainingJob {

  /**
   * Create the weka classifier training job
   *
   * @param builder
   *          the builder
   */
  _WekaClassifierTrainingJob(final ClassifierTrainingJobBuilder builder) {
    super(builder);
  }

  /**
   * do the training.
   *
   * @param instances
   *          the instances
   * @return the classifier
   */
  abstract CT _train(Instances instances);

  /**
   * Create the weka classifier wrapper
   *
   * @param classifier
   *          the classifier
   * @param vector
   *          the attribute vector
   * @param instance
   *          to use
   * @return the weka classifier wrapper
   */
  abstract _WekaClassifier<CT> _createClassifier(final CT classifier,
      final double[] vector, final Instance instance);

  /** {@iheritDoc} */
  @SuppressWarnings({ "rawtypes", "unchecked" })
  @Override
  protected final IClassifierTrainingResult doCall()
      throws IllegalArgumentException {
    ArrayList<Attribute> attributes;
    ArrayList<String> values;
    Instances instances;
    String name;
    int index, index2, max, current;
    double[] vector;
    IClassifier classifier;
    Instance used;
    Object token;
    ClassifierTrainingResult result;
    double quality;
    CT wekaClassifier;

    attributes = new ArrayList<>();
    index = 0;
    values = null;
    name = null;

    // Create the attribute sets.
    for (index = 0; index <= this.m_featureTypes.length; index++) {
      name = Integer.toString(index, Character.MAX_RADIX);

      if (index < this.m_featureTypes.length) {
        // OK, this is a normal attribute
        if (this.m_featureTypes[index] == EFeatureType.NUMERICAL) {
          // Numerical attributes just need a name, nothing else
          attributes.add(new Attribute(name));
          continue;
        }

        // Find the number of different values of the binary or nominal
        // attribute. Such attributes are index values starting at zero. If
        // the maximum value is "max", then there are values like 0...max,
        // i.e., max+1 in total.
        max = 0;
        for (final ClassifiedSample sample : this.m_knownSamples) {
          current = ((int) (sample.featureValues[index]));
          if (current > max) {
            max = current;
          }
        }
      } else {
        // OK, we have class attribute: This is basically a nominal
        // attribute, with values starting at 0 and going to max,
        // indicating max+1 classes.
        max = 0;
        for (final ClassifiedSample sample : this.m_knownSamples) {
          if (sample.sampleClass > max) {
            max = sample.sampleClass;
          }
        }
      }

      // find create the list
      if (values == null) {
        values = new ArrayList<>();
      }

      for (index2 = values.size(); index2 <= max; index2++) {
        values.add(Integer.toString(index2, Character.MAX_RADIX));
      }

      ++max;
      attributes.add(new Attribute(name,
          (values.size() != max) ? values.subList(0, max) : values));
    }

    // Now build the instances set.
    instances = new Instances(name, attributes,
        this.m_knownSamples.length);
    vector = null;
    for (final ClassifiedSample sample : this.m_knownSamples) {
      vector = new double[sample.featureValues.length];
      System.arraycopy(sample.featureValues, 0, vector, 0, vector.length);
      vector[vector.length - 1] = sample.sampleClass;
      instances.add(new DenseInstance(1d, vector));
    }

    // Having built all the data, we can train the classifier.
    wekaClassifier = this._train(instances);

    used = instances.remove(instances.size() - 1);
    instances.classAttribute();
    instances.add(used.copy(vector));
    used = instances.get(0);
    classifier = this._createClassifier(wekaClassifier, vector, used);

    // Evaluate and return the classifier.

    this.m_featureTypes = null;

    token = this.m_qualityMeasure.createToken(this.m_knownSamples);
    quality = ((IClassifierQualityMeasure) (this.m_qualityMeasure))
        .evaluate(classifier, token, this.m_knownSamples);
    token = null;
    this.m_knownSamples = null;
    this.m_qualityMeasure = null;

    result = new ClassifierTrainingResult(classifier, quality);
    classifier = null;

    return result;
  }
}
