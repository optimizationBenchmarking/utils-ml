package org.optimizationBenchmarking.utils.ml.classification.impl.multi;

import java.util.concurrent.Callable;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierQualityMeasure;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;

/** the classifier training job */
final class _ClassifierTrainingTestJob implements Callable<Double> {

  /** the logger to use */
  private final Logger m_logger;
  /** the feature types */
  private final EFeatureType[] m_featureTypes;
  /** the trainer to use */
  private final IClassifierTrainer m_trainer;
  /** the measure */
  private final IClassifierQualityMeasure<?> m_measure;
  /** the data sample */
  private final ClassifiedSample[][] m_testTrain;

  /**
   * create the training-test job
   *
   * @param logger
   *          the logger to use
   * @param featureTypes
   *          the feature types
   * @param trainer
   *          the trainer
   * @param measure
   *          the measure
   * @param testTrain
   *          the test and training set
   */
  _ClassifierTrainingTestJob(final Logger logger,
      final EFeatureType[] featureTypes, final IClassifierTrainer trainer,
      final IClassifierQualityMeasure<?> measure,
      final ClassifiedSample[][] testTrain) {
    super();

    this.m_logger = logger;
    this.m_featureTypes = featureTypes;
    this.m_trainer = trainer;
    this.m_measure = measure;
    this.m_testTrain = testTrain;
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "unchecked", "rawtypes" })
  @Override
  public final Double call() {
    final IClassifierQualityMeasure measure;
    measure = this.m_measure;
    return Double.valueOf(//
        measure.evaluate(//
            this.m_trainer.use().setLogger(this.m_logger)//
                .setFeatureTypes(this.m_featureTypes)//
                .setQualityMeasure(measure)//
                .setTrainingSamples(this.m_testTrain[1])//
                .create().call().getClassifier(), //
            measure.createToken(this.m_testTrain[0]), //
            this.m_testTrain[0]));

  }
}