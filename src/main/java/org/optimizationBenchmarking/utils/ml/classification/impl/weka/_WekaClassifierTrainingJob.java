package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import java.util.ArrayList;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingResult;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.SimplifyingClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * a classifier training job wrapping around Weka
 *
 * @param <CT>
 *          the classifier type
 */
abstract class _WekaClassifierTrainingJob<CT extends Classifier>
    extends SimplifyingClassifierTrainingJob {

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
   * @param instance
   *          to use
   * @return the weka classifier wrapper
   */
  abstract _WekaClassifier<CT> _createClassifier(final CT classifier,
      final _InternalInstance instance);

  /**
   * convert an integer to a name
   *
   * @param index
   *          the integer
   * @return the name
   */
  private static final String __name(final int index) {
    return Integer.toString(index, Character.MAX_RADIX);
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "rawtypes", "unchecked" })
  @Override
  protected final IClassifierTrainingResult doCall()
      throws IllegalArgumentException {
    ArrayList<Attribute> features;
    ArrayList<String> baseValues;
    Instances instances;
    String name;
    int index, index2, max, current;
    IClassifier classifier;
    Object token;
    ClassifierTrainingResult result;
    double quality;
    CT wekaClassifier;
    String[] values;

    features = new ArrayList<>();
    index = 0;
    name = null;
    baseValues = new ArrayList<>();

    // Create the feature sets.
    for (index = 0; index <= this.m_selectedFeatures.length; index++) {

      for (index2 = baseValues.size(); index2 <= index; index2++) {
        baseValues.add(_WekaClassifierTrainingJob.__name(index2));
      }
      name = baseValues.get(index);

      if (index < this.m_selectedFeatures.length) {
        // OK, this is a normal feature
        if (this.m_featureTypes[this.m_selectedFeatures[index]] == EFeatureType.NUMERICAL) {
          // Numerical features just need a name, nothing else
          features.add(new Attribute(name));
          continue;
        }

        // Find the number of different values of the binary or nominal
        // feature. Such features are index values starting at zero. If
        // the maximum value is "max", then there are values like 0...max,
        // i.e., max+1 in total.
        max = 0;
        for (final ClassifiedSample sample : this.m_knownSamples) {
          current = ClassificationTools.featureDoubleToNominal(
              sample.featureValues[this.m_selectedFeatures[index]]);
          if (current > max) {
            max = current;
          }
        }
      } else {
        // OK, we have class feature: This is basically a nominal
        // feature, with values starting at 0 and going to max,
        // indicating max+1 classes.
        max = 0;
        for (final ClassifiedSample sample : this.m_knownSamples) {
          if (sample.sampleClass > max) {
            max = sample.sampleClass;
          }
        }
      }

      // find create the list
      values = new String[max + 1];
      current = baseValues.size();
      for (index2 = baseValues.size(); index2 <= max; index2++) {
        baseValues.add(
            values[index2] = _WekaClassifierTrainingJob.__name(index2));
      }
      for (index2 = Math.min(current, values.length); (--index2) >= 0;) {
        values[index2] = baseValues.get(index2);
      }

      features.add(new Attribute(name, new ArrayListView<>(values)));
    }

    // Now build the instances set.
    instances = new Instances(name, features, this.m_knownSamples.length);
    instances.setClassIndex(this.m_selectedFeatures.length);
    for (final ClassifiedSample sample : this.m_knownSamples) {
      instances.add(//
          new _InternalInstance(sample, this.m_selectedFeatures));
    }

    // Having built all the data, we can train the classifier.
    wekaClassifier = this._train(instances);
    instances.clear();

    instances.add(new _InternalInstance(this.m_selectedFeatures.length));
    classifier = this._createClassifier(wekaClassifier,
        ((_InternalInstance) (instances.get(0))));

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
