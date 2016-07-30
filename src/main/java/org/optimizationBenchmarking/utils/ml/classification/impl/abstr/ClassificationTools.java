package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.math.combinatorics.Shuffle;
import org.optimizationBenchmarking.utils.math.functions.numeric.CeilDiv;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;

/** Some simple tools for classification. */
public final class ClassificationTools {

  /** the default number of cross validation elements */
  private static final int DEFAULT_CROSSVALIDATION_FOLDS = 10;

  /**
   * Can we cluster the given samples trivially? If so, return a job which
   * just directly returns a fixed result. Otherwise return {@code null}.
   *
   * @param samples
   *          the samples
   * @return the job if classification can be done trivially, or
   *         {@code null} if trivial classifying is not possible
   */
  public static final IClassifierTrainingJob canClusterTrivially(
      final ClassifiedSample[] samples) {
    int minClass, maxClass;

    // check whether there is more than one class
    maxClass = (-1);
    minClass = Integer.MAX_VALUE;
    for (final ClassifiedSample sample : samples) {
      if (sample.sampleClass < minClass) {
        minClass = sample.sampleClass;
      }
      if (sample.sampleClass > maxClass) {
        maxClass = sample.sampleClass;
      }
    }

    if (minClass >= maxClass) {
      return new _AllTheSameClass(minClass);
    }

    return null;
  }

  /**
   * Can we cluster the given samples trivially? If so, return a job which
   * just directly returns a fixed result. Otherwise return {@code null}.
   *
   * @param samples
   *          the samples
   * @return the job if classification can be done trivially, or
   *         {@code null} if trivial classifying is not possible
   */
  public static final int getClassCount(final ClassifiedSample[] samples) {

    int maxClass;

    maxClass = (-1);
    for (final ClassifiedSample sample : samples) {
      if (sample.sampleClass > maxClass) {
        maxClass = sample.sampleClass;
      }
    }

    return (maxClass + 1);
  }

  /**
   * Divide the data sample for cross validation. This will create an array
   * of cross-validation tuples, where the first element of each tuple is a
   * test set and the second element is a training set. This goal of this
   * method is to achieve that all test sets contain samples from all
   * classes and that the proportion the classes is about the same in the
   * test sets as in the population (which thus also holds for the training
   * sets). If all classes appear sufficiently often, this method will
   * return {@value #DEFAULT_CROSSVALIDATION_FOLDS} such tuples. However,
   * there may be cases where more tuples are generated, specifically if
   * there is one class with at least two but less than
   * {@value #DEFAULT_CROSSVALIDATION_FOLDS} samples. If there is a class
   * with less than two samples or all samples belong to the same class,
   * cross validation makes no sense and {@code null} is returned.
   *
   * @param samples
   *          the sample
   * @param random
   *          the random number generator
   * @return the cross-validation samples, or {@code null} if no meaningful
   *         division could be created
   */
  public static final ClassifiedSample[][][] divideForCrossValidation(
      final ClassifiedSample[] samples, final Random random) {
    final ClassifiedSample[][] perClass;
    final ClassifiedSample[][][] result;
    ClassifiedSample[] classSamples;
    ArrayList<ClassifiedSample>[] finder;
    ArrayList<ClassifiedSample> current;
    final int[] used;
    final int classCount, feasibleDivisions, crossValidationDivisions;
    int currentClass, currentValidation, nextPick, size, smallestClass,
        index, index2, add;

    // We first find the number of classes.
    classCount = ClassificationTools.getClassCount(samples);

    // Cross-validation makes no sense on one class.
    if (classCount <= 1) {
      return null;
    }

    // Now we separate the samples according to their classes and find the
    // smallest class. Afterwards, the array perClass has length classCount
    // and, for each class, holds another array with the samples belonging
    // to that class. Also, we know the class "smallestClass" with the
    // fewest members.
    finder = new ArrayList[classCount];
    for (final ClassifiedSample sample : samples) {
      current = finder[sample.sampleClass];
      if (current == null) {
        finder[sample.sampleClass] = current = new ArrayList<>();
      }
      current.add(sample);
    }

    perClass = new ClassifiedSample[classCount][];
    smallestClass = Integer.MAX_VALUE;
    for (currentClass = classCount; (--currentClass) >= 0;) {
      current = finder[currentClass];
      if (current != null) {
        size = current.size();
        perClass[currentClass] = current.toArray(//
            new ClassifiedSample[size]);
        if (size < smallestClass) {
          smallestClass = size;

          // If the smallest class has only 1 instance, cross validation
          // makes no sense.
          if (smallestClass <= 1) {
            return null;
          }
        }
      }
    }

    // Ok, we have now the classes separated and can create "fair" folds:
    // Each fold should reflect the overall population, meaning that the
    // number of samples from class A in the fold should be proportional to
    // the number samples from A in the overall population, for any class A
    // - at least approximately. This means that if the smallest class has
    // 6 members, we cannot have more than 6 folds. Since we usually want
    // to have at least 10 folds, we will then just create two times 6
    // folds.
    if (smallestClass < ClassificationTools.DEFAULT_CROSSVALIDATION_FOLDS) {
      feasibleDivisions = smallestClass;
      crossValidationDivisions = CeilDiv.INSTANCE.computeAsInt(
          ClassificationTools.DEFAULT_CROSSVALIDATION_FOLDS,
          feasibleDivisions) * feasibleDivisions;
    } else {
      feasibleDivisions = crossValidationDivisions = ClassificationTools.DEFAULT_CROSSVALIDATION_FOLDS;
    }

    // We now begin sampling the classes and therefore first allocate the
    // necessary objects.
    result = new ClassifiedSample[crossValidationDivisions][2][];
    used = new int[classCount];

    finder = Arrays.copyOf(finder, feasibleDivisions);
    for (index = feasibleDivisions; (--index) >= 0;) {
      if (finder[index] == null) {
        finder[index] = new ArrayList<>();
      }
    }

    // The main loop generates the folds.
    for (currentValidation = 0; currentValidation < crossValidationDivisions; currentValidation += feasibleDivisions) {

      // First, we randomize all samples on a per-class basis.
      for (final ClassifiedSample[] clazz : perClass) {
        Shuffle.shuffle(clazz, random);
      }
      Arrays.fill(used, 0);

      // Now we generate the test samples for each fold. these will be
      // mutually exclusive.

      // Therefore, we fill all the samples with a fair share of each
      // class. This share is basically the number of samples belonging to
      // the class, divided by the number of folds we will create.
      for (final ArrayList<ClassifiedSample> currentDivision : finder) {
        currentDivision.clear();
        for (currentClass = classCount; (--currentClass) >= 0;) {
          classSamples = perClass[currentClass];
          for (add = classSamples.length
              / feasibleDivisions; (--add) >= 0;) {
            currentDivision.add(classSamples[used[currentClass]++]);
          }
        }
      }

      // Now we fairly distribute remaining items: Dividing the class sizes
      // by the number of folds may yield a remainder, so these elements
      // are iteratively assigned to folds as well.
      nextPick = 0;
      for (currentClass = classCount; (--currentClass) >= 0;) {
        classSamples = perClass[currentClass];
        while (used[currentClass] < classSamples.length) {
          finder[nextPick].add(classSamples[used[currentClass]++]);
          nextPick = ((nextPick + 1) % feasibleDivisions);
        }
      }

      // Store the validation/test sets.
      for (index = feasibleDivisions; (--index) >= 0;) {
        current = finder[index];
        result[currentValidation + index][0] = current
            .toArray(new ClassifiedSample[current.size()]);
      }
      current = finder[0];// avoid compiler warning

      // Now generate the training sets.
      for (index = feasibleDivisions; (--index) >= 0;) {
        current.clear();
        for (index2 = feasibleDivisions; (--index2) >= 0;) {
          if (index2 == index) {
            continue;
          }
          for (final ClassifiedSample sample : result[currentValidation
              + index2][0]) {
            current.add(sample);
          }
        }
        // Store he training set.
        result[currentValidation + index][1] = current
            .toArray(new ClassifiedSample[current.size()]);
      }
    }

    return result;
  }

  /** do not call */
  private ClassificationTools() {
    ErrorUtils.doNotCall();
  }
}
