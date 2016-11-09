package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.optimizationBenchmarking.utils.comparison.EComparison;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.math.combinatorics.Shuffle;
import org.optimizationBenchmarking.utils.math.functions.arithmetic.AddN;
import org.optimizationBenchmarking.utils.math.functions.numeric.CeilDiv;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.text.TextUtils;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** Some simple tools for classification. */
public final class ClassificationTools {

  /** the text for an {@code if}, followed by a space */
  public static final char[] RULE_IF = { 'i', 'f', ' ' };
  /** the text for an {@code else if}, followed by a space */
  public static final char[] RULE_ELSE_IF = { 'e', 'l', 's', 'e', ' ', 'i',
      'f', ' ' };
  /** the text for an {@code else}, followed by a space */
  public static final char[] RULE_ELSE = { 'e', 'l', 's', 'e', ' ' };
  /** the text for an {@code then}, preceeded by a space */
  public static final char[] RULE_THEN = { ' ', 't', 'h', 'e', 'n' };
  /** the text for a condition which is always true */
  public static final char[] RULE_ALWAYS_TRUE = { 't', 'r', 'u', 'e' };

  /** the default number of cross validation elements */
  private static final int DEFAULT_CROSSVALIDATION_FOLDS = 10;

  /** the basic unit of complexity */
  private static final double COMPLEXITY_UNIT = 1d;

  /** the complexity unit for depth */
  public static final double COMPLEXITY_DEPTH_UNIT = (ClassificationTools.COMPLEXITY_UNIT
      / 32d);
  /** the complexity unit for comparisons */
  public static final double COMPLEXITY_COMPARISON_UNIT = (ClassificationTools.COMPLEXITY_UNIT
      / 4d);
  /** the complexity unit for arithmetic operations */
  public static final double COMPLEXITY_ARITHMETIC_UNIT = (ClassificationTools.COMPLEXITY_UNIT
      / 2d);
  /** the complexity unit for logic operations */
  public static final double COMPLEXITY_LOGIC_UNIT = (ClassificationTools.COMPLEXITY_ARITHMETIC_UNIT
      / 2d);
  /** the complexity unit for features */
  public static final double COMPLEXITY_FEATURE_UNIT = (ClassificationTools.COMPLEXITY_UNIT);
  /** the complexity unit for constants */
  public static final double COMPLEXITY_CONSTANT_UNIT = (ClassificationTools.COMPLEXITY_UNIT
      / 8d);
  /** the complexity unit for classes */
  public static final double COMPLEXITY_CLASS_UNIT = (ClassificationTools.COMPLEXITY_UNIT);
  /** the complexity unit for if-then-or-else decisions */
  public static final double COMPLEXITY_DECISION_UNIT = (ClassificationTools.COMPLEXITY_UNIT
      / 4d);

  /** The text for unspecified features */
  public static final char[] FEATURE_IS_UNSPECIFIED = { 'u', 'n', 's', 'p',
      'e', 'c', 'i', 'f', 'i', 'e', 'd' };
  /** The text for specified features */
  public static final char[] FEATURE_IS_SPECIFIED = { 's', 'p', 'e', 'c',
      'i', 'f', 'i', 'e', 'd' };

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

    if ((samples == null) || (samples.length <= 0)) {
      return new _AllTheSameClass(0);
    }

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
   * <p>
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
   * </p>
   * <p>
   * TODO: This should rely only on {@link ClassifiedSampleInfo} for
   * statistics. Right now, it will only work on top-level jobs where all
   * classes appear.
   * </p>
   *
   * @param samples
   *          the sample
   * @param random
   *          the random number generator
   * @return the cross-validation samples, or {@code null} if no meaningful
   *         division could be created
   */
  @SuppressWarnings("unchecked")
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
    classCount = new ClassifiedSampleInfo(samples).getBiggestSampleClass();

    // Cross-validation makes no sense on one class or if we have less
    // samples than two times the number of folds, i.e., less than 20
    // samples.
    if ((classCount <= 1) || (samples.length < (2
        * ClassificationTools.DEFAULT_CROSSVALIDATION_FOLDS))) {
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

  /**
   * Check a classifier training result record and throw a
   * {@link IllegalArgumentException} if the result is invalid.
   *
   * @param result
   *          the result record
   */
  public static final void checkClassifierTrainingResult(
      final IClassifierTrainingResult result) {
    final double quality;

    if (result == null) {
      throw new IllegalArgumentException(
          "Result returned by classifier trainer cannot be null.");//$NON-NLS-1$
    }
    if (result.getClassifier() == null) {
      throw new IllegalArgumentException(
          "Classifier stored in training result cannot be null.");//$NON-NLS-1$
    }

    quality = result.getQuality();
    if ((quality < 0d) || (quality != quality)) {
      throw new IllegalArgumentException(
          "Invalid classifier quality:" + quality); //$NON-NLS-1$
    }
  }

  /**
   * Print the name of a class
   *
   * @param clazz
   *          the class
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  public static final void printClass(final int clazz,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append("class = "); //$NON-NLS-1$
    renderer.renderShortClassName(clazz, textOutput);
  }

  /**
   * Print an expression which compares a the value of an feature with a
   * specified value.
   *
   * @param feature
   *          the feature
   * @param comparison
   *          the comparison
   * @param value
   *          the value to compare with
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  @SuppressWarnings("incomplete-switch")
  public static final void printFeatureExpression(final int feature,
      final EComparison comparison, final double value,
      final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    final int specifiedSwitch;

    try {
      renderer.renderShortFeatureName(feature, textOutput);

      findSpec: {
        if (EFeatureType.featureDoubleIsUnspecified(value)) {
          switch (comparison) {
            case EQUAL: {
              specifiedSwitch = -1;
              break findSpec;
            }
            case NOT_EQUAL: {
              specifiedSwitch = 1;
              break findSpec;
            }
            default: {
              throw new IllegalArgumentException("The " + feature //$NON-NLS-1$
                  + "th feature is unspecified and the comparison to be used in the expression is " //$NON-NLS-1$
                  + comparison
                  + ", but only EQUAL and NOT_EQUAL are permitted for unspecified features."); //$NON-NLS-1$
            }
          }
        }

        // try to deal with odd situations where a feature seems to be
        // specified via a "back-door"
        comp: switch (comparison) {
          case LESS:
          case LESS_OR_EQUAL: {
            if (value >= Double.MAX_VALUE) {
              specifiedSwitch = 1;
              break findSpec;
            }
            if (value <= (-Double.MAX_VALUE)) {
              specifiedSwitch = -1;
              break findSpec;
            }
            break comp;
          }

          case GREATER:
          case GREATER_OR_EQUAL: {
            if (value >= Double.MAX_VALUE) {
              specifiedSwitch = -1;
              break findSpec;
            }
            if (value <= (-Double.MAX_VALUE)) {
              specifiedSwitch = 1;
              break findSpec;
            }
            break comp;
          }
        }

        specifiedSwitch = 0;
      }

      switch (specifiedSwitch) {
        case -1: {
          textOutput.append(" is "); //$NON-NLS-1$
          textOutput.append(ClassificationTools.FEATURE_IS_UNSPECIFIED);
          return;
        }
        case 1: {
          textOutput.append(" is "); //$NON-NLS-1$
          textOutput.append(ClassificationTools.FEATURE_IS_SPECIFIED);
          return;
        }
      }

      textOutput.append(' ');

      switch (comparison) {
        case LESS: {
          textOutput.append('<');
          break;
        }
        case LESS_OR_EQUAL: {
          textOutput.append('<');
          textOutput.append('=');
          break;
        }
        case EQUAL: {
          textOutput.append('=');
          break;
        }
        case GREATER_OR_EQUAL: {
          textOutput.append('>');
          textOutput.append('=');
          break;
        }
        case GREATER: {
          textOutput.append('>');
          break;
        }
        case NOT_EQUAL: {
          textOutput.append('!');
          textOutput.append('=');
          break;
        }
        default: {
          throw new IllegalArgumentException(
              "Unsupported comparison: " + comparison); //$NON-NLS-1$
        }
      }

      textOutput.append(' ');
      renderer.renderFeatureValue(feature, value, textOutput);
    } catch (final Throwable error) {
      throw new IllegalArgumentException(//
          "Error when rendering value " + value //$NON-NLS-1$
              + " for feature " + feature + //$NON-NLS-1$
              " and comparison " + comparison + //$NON-NLS-1$
              " for renderer " + TextUtils.className(renderer) //$NON-NLS-1$
              + " to output " + TextUtils.className(textOutput), //$NON-NLS-1$
          error);
    }
  }

  /**
   * Combine several nested complexities
   *
   * @param nested
   *          the list of nested complexities
   * @return the result
   */
  public static final double complexityNested(final double... nested) {
    double[] add;

    add = new double[nested.length + 1];
    System.arraycopy(nested, 0, add, 0, nested.length);
    add[nested.length] = ClassificationTools.COMPLEXITY_DEPTH_UNIT;
    return AddN.destructiveSum(add);
  }

  /** do not call */
  private ClassificationTools() {
    ErrorUtils.doNotCall();
  }
}
