package org.optimizationBenchmarking.utils.ml.classification.impl.multi;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.math.statistics.aggregate.QuantileAggregate;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassificationTools;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.parallel.Execute;

/**
 * <p>
 * A training job for multiple classifier.
 * </p>
 * <p>
 * In a multi-classifier training job, there may be multiple classifier
 * trainers available. We want to choose the best classifier. In the ideal
 * case, we can do this by first doing cross validation. Then, we will
 * first apply the available trainers to server tuples of training and test
 * sets. We then choose the trainer with the best median performance on the
 * test training sets. Of course, this could be more than one. This means,
 * we can have a set of classifier trainers: either because we did
 * cross-validation and multiple trainers have the same median test
 * performance or we could not do cross-validation, because we had too few
 * samples. We will take the list of (hopefully best) classifier trainers
 * and apply them to the whole data set. Among the results, we pick the one
 * with the best quality rating.
 * </p>
 */
final class _MultiClassifierTrainingJob extends ClassifierTrainingJob {

  /** the trainers to be used */
  private Collection<IClassifierTrainer> m_trainers;

  /**
   * Create the job
   *
   * @param builder
   *          the job builder
   */
  _MultiClassifierTrainingJob(
      final MultiClassifierTrainingJobBuilder builder) {
    super(builder);
    MultiClassifierTrainingJobBuilder
        ._checkTrainers(this.m_trainers = builder.m_trainers);
  }

  /** {@inheritDoc} */
  @SuppressWarnings("unchecked")
  @Override
  protected final IClassifierTrainingResult doCall() {
    final int classifierCount, foldCount;
    final Logger logger;
    Collection<IClassifierTrainer> trainers;
    ArrayList<IClassifierTrainer> useTrainers;
    ArrayList<Throwable> errors;
    ClassifiedSample[][][] folds;
    Future<Double>[] crossValidationTestQualities;
    Future<IClassifierTrainingResult>[] results;
    double bestMedian, currentMedian, bestQuality, currentQuality;
    QuantileAggregate median;
    int index, index2;
    IClassifierTrainingResult bestResult, currentResult;
    String message;
    IllegalStateException finalError;

    trainers = this.m_trainers;
    this.m_trainers = null;

    classifierCount = trainers.size();
    logger = this.getLogger();
    errors = null;

    // In a multi-classifier training job, there may be multiple classifier
    // trainers available. We want to choose the best classifier. In the
    // ideal case, we can do this by first doing cross validation. Then, we
    // will first apply the available trainers to server tuples of training
    // and test sets. We then choose the trainer with the best median
    // performance on the test training sets. Of course, this could be more
    // than one. This means, we can have a set of classifier trainers:
    // either because we did cross-validation and multiple trainers have
    // the same median test performance or we could not do
    // cross-validation, because we had too few samples. We will take the
    // list of (hopefully best) classifier trainers and apply them to the
    // whole data set. Among the results, we pick the one with the best
    // quality rating.
    if (classifierCount > 1) {

      // If there is more than one trainer, we need to choose the best one
      folds = ClassificationTools.divideForCrossValidation(
          this.m_knownSamples, ThreadLocalRandom.current());

      if ((folds != null) && ((foldCount = folds.length) > 1)) {
        if ((logger != null) && (logger.isLoggable(Level.FINE))) {
          logger.fine(
              "Multi-Classifier-Training job will now determine the best of the " //$NON-NLS-1$
                  + classifierCount + " classification trainers " //$NON-NLS-1$
                  + trainers + " using the median test quality on " //$NON-NLS-1$
                  + foldCount + "-fold stratified cross-validation."); //$NON-NLS-1$
        }

        // Fork of the parallel jobs: For each "fold", first train a
        // classifier, then test it. Do this for each trainer.
        crossValidationTestQualities = new Future[classifierCount
            * folds.length];
        index = 0;
        for (final IClassifierTrainer trainer : trainers) {
          for (final ClassifiedSample[][] fold : folds) {
            crossValidationTestQualities[index++] = Execute.parallel(
                new _ClassifierTrainingTestJob(logger, this.m_featureTypes,
                    trainer, this.m_qualityMeasure, fold));
          }
        }

        // OK, so we have forked off all the training/test jobs

        // If we can use cross-validation, we will compute the median
        // quality of the each trainer on the test sets belonging to the
        median = new QuantileAggregate(0.5d);
        useTrainers = new ArrayList<>();
        index = 0;
        bestMedian = Double.POSITIVE_INFINITY;
        for (final IClassifierTrainer trainer : trainers) {
          median.reset();

          // Append all the values to the median aggregate of the current
          // trainer.
          for (index2 = foldCount; (--index2) >= 0;) {
            try {
              median.append(crossValidationTestQualities[index].get());
              crossValidationTestQualities[index++] = null;
            } catch (final Throwable error) {
              // if something goes wrong aggregate errors
              if (errors == null) {
                errors = new ArrayList<>();
              }
              errors.add(error);
            }
          }

          // OK, we have computed the median, so we can now check if the
          // median is sufficiently good.
          currentMedian = median.doubleValue();
          if ((useTrainers.isEmpty()) || ((currentMedian >= 0)
              && (currentMedian < Double.POSITIVE_INFINITY))) {
            add: {
              if (currentMedian > bestMedian) {
                // current classifier trainer is worse
                break add;
              }
              if (currentMedian < bestMedian) {
                // new best: drop all collected trainers
                bestMedian = currentMedian;
                useTrainers.clear();
              }
              // current trainer better or equally good to best, remember
              useTrainers.add(trainer);
            }
          }
        }
        crossValidationTestQualities = null;

        // Based on these results, we can now pick the classifiers. We
        // choose the classifiers with the best median test quality. This
        // could be more than one. We also print the description of this
        // process to the log.
        if (useTrainers.isEmpty()) {
          if ((logger != null) && (logger.isLoggable(Level.WARNING))) {
            message = ("Something went wrong when trying to pick the best of the " //$NON-NLS-1$
                + classifierCount + " trainers "//$NON-NLS-1$
                + trainers + " on " + //$NON-NLS-1$
                foldCount + "-fold stratified cross-validation."); //$NON-NLS-1$
            if (errors != null) {
              logger.log(Level.WARNING, message, errors);
            } else {
              logger.warning(message);
            }
          } else {
            if (useTrainers.size() < classifierCount) {
              trainers = useTrainers;
              if ((logger != null) && (logger.isLoggable(Level.FINE))) {
                logger.fine(//
                    "The cross-validation helped us to reduce the number of candidate classifiers from "//$NON-NLS-1$
                        + classifierCount + " to " + useTrainers.size() + //$NON-NLS-1$
                        ", namely " + useTrainers + //$NON-NLS-1$
                        ", which all have the same median test quality"//$NON-NLS-1$
                        + bestMedian);
              }
            } else {
              if ((logger != null) && (logger.isLoggable(Level.FINE))) {
                logger.fine(//
                    "Cross-validation did not help us to distinguish the classifiers, they all have the same median test quality "//$NON-NLS-1$
                        + bestMedian);
              }
            }

          }
        }
        useTrainers = null;
      } else {
        // Cross validation not possible. Maybe we have too few samples of
        // some class.
        if ((logger != null) && (logger.isLoggable(Level.FINE))) {
          logger.fine("Multi-Classifier-Training job will apply all " //$NON-NLS-1$
              + classifierCount
              + " classification trainers on the whole data set and pick the result with best quality."); //$NON-NLS-1$
        }
      }
    } else {
      // There was only one trainer anyway, so we don't even need to
      // attempt cross-validation to pick the best one.
      if ((logger != null) && (logger.isLoggable(Level.FINE))) {
        logger.fine(
            "Multi-Classifier-Training only has one classifier trainer, so we use it directly."); //$NON-NLS-1$
      }
    }

    // Now train the classifiers on the full data set: Fork off one task
    // for each trainer.
    results = new Future[trainers.size()];
    index = 0;
    for (final IClassifierTrainer trainer : trainers) {
      results[index++] = Execute.parallel(//
          trainer.use()//
              .setLogger(logger)//
              .setFeatureTypes(this.m_featureTypes)//
              .setQualityMeasure(this.m_qualityMeasure)//
              .setTrainingSamples(this.m_knownSamples)//
              .create());
    }
    trainers = null;
    this.m_trainers = null;
    this.m_featureTypes = null;
    this.m_knownSamples = null;
    this.m_qualityMeasure = null;

    // OK, all the tasks are running, we can now reap the results.
    bestResult = null;
    bestQuality = Double.POSITIVE_INFINITY;
    for (index = 0; index < results.length; index++) {
      try {
        currentResult = results[index].get();
        results[index] = null;
        currentQuality = currentResult.getQuality();
        if ((bestResult == null) || ((currentQuality >= 0d)
            && (currentQuality < bestQuality))) {
          bestQuality = currentQuality;
          bestResult = currentResult;
        }
      } catch (final Throwable error) {
        if (errors == null) {
          errors = new ArrayList<>();
        }
        errors.add(error);
      }
    }

    if (bestResult != null) {
      return bestResult;
    }

    // Ok, not a single classifier could successfully train on the data. We
    // try to throw a descriptive error.
    message = "Errors occured during all " + results//$NON-NLS-1$
        + " fitting procedures on the training data.";//$NON-NLS-1$
    results = null;
    if (errors != null) {
      index = errors.size();
      if (index > 0) {
        if (index == 1) {
          throw new IllegalStateException(message, errors.get(0));
        }
        finalError = new IllegalStateException(message);
        for (final Throwable error : errors) {
          finalError.addSuppressed(error);
        }
        throw finalError;
      }
    }

    throw new IllegalStateException(message);
  }

}
