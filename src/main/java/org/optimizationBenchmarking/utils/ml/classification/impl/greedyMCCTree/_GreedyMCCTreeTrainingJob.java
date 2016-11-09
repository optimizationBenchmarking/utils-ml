package org.optimizationBenchmarking.utils.ml.classification.impl.greedyMCCTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;

import org.optimizationBenchmarking.utils.math.functions.power.Log;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ClassifierTrainingResult;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.ConfusionMatrix;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.SimplifyingClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.impl.quality.MCC;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.EFeatureType;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;

/** A training job for greedy tree-based classifiers */
final class _GreedyMCCTreeTrainingJob
    extends SimplifyingClassifierTrainingJob {

  /**
   * Create the weka classifier training job
   *
   * @param builder
   *          the builder
   */
  _GreedyMCCTreeTrainingJob(final ClassifierTrainingJobBuilder builder) {
    super(builder);
  }

  /**
   * train a classifier node, recursively
   *
   * @param samples
   *          the samples
   * @param types
   *          the types
   * @param mcc
   *          the measure
   * @param skip
   *          the attribute to skip
   * @param maxIntervals
   *          the maximum number of real intervals
   * @param maxDepth
   *          the maximum depth
   * @return the node
   */
  private static final ClassifierTrainingResult __train(
      final ClassifiedSample[] samples, final EFeatureType[] types,
      final MCC mcc, final int skip, final int maxIntervals,
      final int maxDepth) {
    ConfusionMatrix matrix;
    _GreedyMCCTree node;
    ClassifierTrainingResult bestResult, testResult;
    int index;

    matrix = mcc.createToken(samples);
    node = new _GreedyMCCTreeLeaf(matrix.getBiggestSampleClass());
    bestResult = new ClassifierTrainingResult(node,
        mcc.evaluate(node, matrix, samples), node._complexity());

    if (maxDepth > 0) {
      for (index = 0; index < types.length; index++) {
        if (index == skip) {
          continue;
        }
        testResult = _GreedyMCCTreeTrainingJob.__trainForAttribute(index,
            samples, types, matrix, mcc, maxIntervals, (maxDepth - 1));
        if (testResult == null) {
          continue;
        }
        if ((testResult.quality < bestResult.quality)
            || ((testResult.quality <= bestResult.quality)
                && (testResult.complexity < bestResult.complexity))) {
          bestResult = testResult;
        }
      }
    }

    return bestResult;
  }

  /**
   * train a classifier node, recursively
   *
   * @param attributeIndex
   *          the attribute index
   * @param samples
   *          the samples
   * @param types
   *          the types
   * @param matrix
   *          the confusion matrix
   * @param mcc
   *          the measure
   * @param maxIntervals
   *          the maximum number of real intervals
   * @param maxDepth
   *          the maximum depth
   * @return the node
   */
  @SuppressWarnings("unchecked")
  private static final ClassifierTrainingResult __trainForAttribute(
      final int attributeIndex, final ClassifiedSample[] samples,
      final EFeatureType[] types, final ConfusionMatrix matrix,
      final MCC mcc, final int maxIntervals, final int maxDepth) {
    HashSet<Double>[] values;
    HashSet<Double> all;
    Double value;
    boolean isNumerical;
    int clazz, found;
    double[][] dvalues;
    _GreedyMCCTree node;
    ArrayList<ClassifiedSample> allSamples;

    if (samples.length <= 1) {
      return null;
    }

    values = new HashSet[matrix.getClassCount()];
    isNumerical = types[attributeIndex] == EFeatureType.NUMERICAL;
    found = 0;
    for (final ClassifiedSample sample : samples) {
      if (isNumerical) {
        value = Double.valueOf(EFeatureType.featureDoubleToNumerical(
            sample.featureValues[attributeIndex]));
      } else {
        value = Double.valueOf(EFeatureType
            .featureDoubleToNominal(sample.featureValues[attributeIndex]));
      }
      clazz = matrix.getClassIndex(sample.sampleClass);
      if (values[clazz] == null) {
        values[clazz] = new HashSet<>();
        ++found;
      }
      values[clazz].add(value);
    }

    // no further distinction is possible
    if (found <= 1) {
      return null;
    }

    dvalues = _GreedyMCCTreeTrainingJob.__setToValues(values);
    values = null;

    all = new HashSet<>(samples.length);
    if (_GreedyMCCTreeTrainingJob.__compact(dvalues, all) <= 1) {
      return null;// one class overlaps the rest entirely
    }

    allSamples = new ArrayList<>(samples.length);
    for (final ClassifiedSample sample : samples) {
      allSamples.add(sample);
    }

    if (isNumerical) {
      node = _GreedyMCCTreeTrainingJob.__trainForNumericalAttribute(
          attributeIndex, allSamples, types, matrix, mcc, dvalues, found,
          all, maxIntervals, maxDepth);
      all = null;
    } else {
      all = null;
      node = _GreedyMCCTreeTrainingJob.__trainForNominalAttribute(
          attributeIndex, allSamples, types, matrix, mcc, dvalues, found,
          maxIntervals, maxDepth);
    }

    if ((node == null) || (node instanceof _GreedyMCCTreeLeaf)) {
      return null;
    }

    return new ClassifierTrainingResult(node,
        mcc.evaluate(node, matrix, samples), node._complexity());
  }

  /**
   * convert a hash set to a set of values
   *
   * @param values
   *          the values
   * @return the array
   */
  private static final double[][] __setToValues(
      final HashSet<Double>[] values) {
    double[][] result;
    int index, index2;
    Object[] array;

    result = new double[values.length][];
    index = -1;
    for (final HashSet<Double> vals : values) {
      ++index;
      if (vals == null) {
        continue;
      }
      array = vals.toArray();
      result[index] = new double[array.length];
      index2 = (-1);
      for (final Object num : array) {
        result[index][++index2] = ((Number) num).doubleValue();
      }
    }

    for (final double[] res : result) {
      Arrays.sort(res);
    }

    return result;
  }

  /**
   * train a classifier node, recursively
   *
   * @param attributeIndex
   *          the attribute index
   * @param samples
   *          the samples
   * @param types
   *          the types
   * @param matrix
   *          the confusion matrix
   * @param mcc
   *          the measure
   * @param values
   *          the values
   * @param found
   *          the number of splits
   * @param all
   *          all the values found so far
   * @param maxIntervals
   *          the maximum number of real intervals
   * @param maxDepth
   *          the maximum depth
   * @return the node
   */
  private static final _GreedyMCCTree __trainForNumericalAttribute(
      final int attributeIndex, final ArrayList<ClassifiedSample> samples,
      final EFeatureType[] types, final ConfusionMatrix matrix,
      final MCC mcc, final double[][] values, final int found,
      final HashSet<Double> all, final int maxIntervals,
      final int maxDepth) {
    final ArrayList<_RangeAssignment> assignments;
    final ArrayList<ClassifiedSample> choice;
    double[][] intervals;
    _RangeAssignment current;
    int size;

    choice = new ArrayList<>(samples.size());
    assignments = new ArrayList<>(found);

    for (final double[] selection : values) {
      if (selection == null) {
        continue;
      }

      intervals = _GreedyMCCTreeTrainingJob.__selectionToArray(selection,
          all, maxIntervals);
      if (intervals == null) {
        continue;
      }

      current = new _RangeAssignment(intervals);

      if (_GreedyMCCTreeTrainingJob.__setClassifier(attributeIndex,
          samples, types, mcc, choice, current, maxIntervals, maxDepth)) {
        assignments.add(current);
      }
    }

    size = assignments.size();
    if (size <= 1) {
      return null;
    }
    return new _GreedyMCCTreeDecisionNode(attributeIndex,
        assignments.toArray(new _Assignment[size]));
  }

  /**
   * transform a selection to an array
   *
   * @param selection
   *          the selection
   * @param remaining
   *          the remaining values
   * @param maxIntervals
   *          the maximum number of real intervals
   * @return the selection
   */
  @SuppressWarnings("null")
  private static final double[][] __selectionToArray(
      final double[] selection, final HashSet<Double> remaining,
      final int maxIntervals) {
    ArrayList<double[]> intervals;
    int current, next, size;
    double currentValue, nextValue, forbiddenValue, nextBigger;
    Double currentDouble;
    double[] currentInterval;
    double[][] array, array2;

    intervals = new ArrayList<>();

    next = -1;
    for (current = 0; current < selection.length; current++) {
      if (current < next) {
        continue;
      }
      currentValue = selection[current];
      currentDouble = Double.valueOf(currentValue);
      if (remaining.remove(currentDouble)) {
        if (EFeatureType.featureDoubleIsUnspecified(currentValue)) {
          currentInterval = null;
        } else {
          currentInterval = new double[] { currentValue,
              nextValue = currentValue, 0d, current };

          // attempt to make interval bigger, but avoid swallowing values
          // coming later
          findNext: for (next = (current
              + 1); next < selection.length; next++) {
            nextValue = selection[next];
            if (EFeatureType.featureDoubleIsUnspecified(nextValue)) {
              break findNext;
            }
            findForbidden: for (final Double forbidden : remaining) {
              forbiddenValue = forbidden.doubleValue();
              if (forbiddenValue == nextValue) {
                continue findForbidden;
              }
              if (_RangeAssignment._check(currentValue, nextValue,
                  forbiddenValue)) {
                break findNext;
              }
            }

            remaining.remove(Double.valueOf(nextValue));

            if (nextValue > currentInterval[1]) {
              currentInterval[1] = nextValue;
            }
          }

          currentInterval[2] = (currentInterval[1] - currentInterval[0]);

          // try to expand to bigger numbers
          nextBigger = Double.POSITIVE_INFINITY;
          for (final Double forbidden : remaining) {
            forbiddenValue = forbidden.doubleValue();
            if ((forbiddenValue > currentInterval[1])
                && (forbiddenValue <= nextBigger)) {
              if (forbiddenValue >= Double.POSITIVE_INFINITY) {
                nextBigger = Double.MAX_VALUE;
              } else {
                nextBigger = forbiddenValue;
              }
            }
          }
          if (nextBigger > currentInterval[1]) {
            if (nextBigger >= Double.POSITIVE_INFINITY) {
              currentInterval[1] = Double.POSITIVE_INFINITY;
            } else {
              currentInterval[1] = Math.max(currentInterval[1],
                  ((0.5d * currentInterval[1]) + (0.5d * nextBigger)));
            }
          }

          // try to expand to smaller numbers
          nextBigger = Double.NEGATIVE_INFINITY;
          for (final Double forbidden : remaining) {
            forbiddenValue = forbidden.doubleValue();
            if ((forbiddenValue < currentInterval[0])
                && (forbiddenValue >= nextBigger)) {
              if (forbiddenValue <= Double.NEGATIVE_INFINITY) {
                nextBigger = (-Double.MAX_VALUE);
              } else {
                nextBigger = forbiddenValue;
              }
            }
          }
          if (nextBigger < currentValue) {
            if (nextBigger <= Double.NEGATIVE_INFINITY) {
              currentInterval[0] = Double.NEGATIVE_INFINITY;
            } else {
              currentInterval[0] = Math.min(currentInterval[0],
                  ((0.5d * currentInterval[0]) + (0.5d * nextBigger)));
            }
          }

        }

        intervals.add(currentInterval);
      }
    }

    size = intervals.size();
    if (size <= 0) {
      return null;
    }

    array = intervals.toArray(new double[size][]);
    switch (array.length) {
      case 0: {
        return null;
      }
      case 1: {
        return new double[][] { //
            { array[0][0], array[0][1] },//
        };
      }
      default: {// very inefficient..
        Arrays.sort(array, new __IntervalComparator(2));
        Arrays.fill(array, 0, (array.length - maxIntervals), null);
        Arrays.sort(array, new __IntervalComparator(maxIntervals));
        array2 = new double[maxIntervals][];
        current = 0;
        for (final double[] vec : intervals) {
          array2[current] = new double[] { vec[0], vec[1] };
          if ((++current) >= maxIntervals) {
            break;
          }
        }
        return array2;
      }
    }
  }

  /**
   * compact the given array
   *
   * @param numbers
   *          the numbers to compact
   * @param done
   *          the destination for discovered values
   * @return the remaining rows
   */
  private static final int __compact(final double[][] numbers,
      final HashSet<Double> done) {
    final ArrayList<Double> current;
    Double cval;
    int rowIndex, rowSize, index, have;

    current = new ArrayList<>();
    rowIndex = (-1);
    have = 0;
    for (final double[] row : numbers) {
      ++rowIndex;
      if (row == null) {
        continue;
      }
      if (row.length <= 0) {
        numbers[rowIndex] = null;
        continue;
      }

      current.clear();
      for (final double val : row) {
        cval = Double.valueOf(val);
        if (done.add(cval)) {
          current.add(cval);
        }
      }

      rowSize = current.size();
      if (row.length <= rowSize) {
        ++have;
        continue;
      }
      if (rowSize <= 0) {
        numbers[rowIndex] = null;
        continue;
      }

      numbers[rowIndex] = new double[rowSize];
      index = (-1);
      for (final Double dbl : current) {
        numbers[rowIndex][++index] = dbl.doubleValue();
      }

      ++have;
    }
    return have;
  }

  /**
   * train a classifier node, recursively
   *
   * @param attributeIndex
   *          the attribute index
   * @param samples
   *          the samples
   * @param types
   *          the types
   * @param matrix
   *          the confusion matrix
   * @param mcc
   *          the measure
   * @param values
   *          the values
   * @param found
   *          the number of splits
   * @param maxIntervals
   *          the maximum number of real intervals
   * @param maxDepth
   *          the maximum depth
   * @return the node
   */
  private static final _GreedyMCCTree __trainForNominalAttribute(
      final int attributeIndex, final ArrayList<ClassifiedSample> samples,
      final EFeatureType[] types, final ConfusionMatrix matrix,
      final MCC mcc, final double[][] values, final int found,
      final int maxIntervals, final int maxDepth) {
    ArrayList<_ListAssignment> assignments;
    ArrayList<ClassifiedSample> choice;
    HashSet<Integer> done;
    ArrayList<Integer> use;
    _ListAssignment current;
    int index, ival;
    Integer iival;

    choice = new ArrayList<>(samples.size());

    done = new HashSet<>();
    use = new ArrayList<>();

    assignments = new ArrayList<>(found);
    for (final double[] value : values) {
      if (value == null) {
        continue;
      }

      use.clear();
      for (final double currentVal : value) {
        ival = ((int) currentVal);
        iival = Integer.valueOf(ival);
        if (done.add(iival)) {
          use.add(iival);
        }
      }

      ival = use.size();
      if (ival <= 0) {
        continue;
      }

      current = new _ListAssignment(new int[ival]);
      index = (-1);
      for (final Integer feature : use) {
        current.m_values[++index] = feature.intValue();
      }
      use.clear();

      if (_GreedyMCCTreeTrainingJob.__setClassifier(attributeIndex,
          samples, types, mcc, choice, current, maxIntervals, maxDepth)) {
        assignments.add(current);
      }
    }

    index = assignments.size();
    if (index <= 1) {
      return null;
    }
    return new _GreedyMCCTreeDecisionNode(attributeIndex,
        assignments.toArray(new _Assignment[index]));
  }

  /**
   * set the classifier of the current assignment
   *
   * @param attributeIndex
   *          the attribute index
   * @param samples
   *          the samples
   * @param types
   *          the types
   * @param mcc
   *          the measure
   * @param choiceDest
   *          the choice destination
   * @param current
   *          the current assignment
   * @param maxIntervals
   *          the maximum number of real intervals
   * @param maxDepth
   *          the maximum depth
   * @return {@code true} on success, {@code false} on failure
   */
  private static final boolean __setClassifier(final int attributeIndex,
      final ArrayList<ClassifiedSample> samples,
      final EFeatureType[] types, final MCC mcc,
      final ArrayList<ClassifiedSample> choiceDest,
      final _Assignment current, final int maxIntervals,
      final int maxDepth) {
    int index;
    ClassifiedSample sample2;
    ClassifierTrainingResult result;

    choiceDest.clear();
    for (index = samples.size(); (--index) >= 0;) {
      sample2 = samples.get(index);
      if (current._check(sample2.featureValues[attributeIndex])) {
        choiceDest.add(sample2);
        samples.remove(index);
      }
    }

    if (choiceDest.isEmpty()) {
      return false;
    }

    result = _GreedyMCCTreeTrainingJob.__train(
        choiceDest.toArray(new ClassifiedSample[choiceDest.size()]), types,
        mcc, attributeIndex, maxIntervals, maxDepth);
    if (result == null) {
      samples.addAll(choiceDest);
      choiceDest.clear();
      return false;
    }
    choiceDest.clear();
    current.m_classifier = ((_GreedyMCCTree) (result.classifier));
    return true;
  }

  /** {@inheritDoc} */
  @Override
  protected final IClassifierTrainingResult doCall() {
    int maxDepth;

    maxDepth = ((int) (Math.round(Log.INSTANCE.computeAsDouble(
        Math.max(2, (this.m_featureTypes.length - 1)), 100000000))));
    return _GreedyMCCTreeTrainingJob.__train(this.m_knownSamples,
        this.m_featureTypes, MCC.INSTANCE, //
        -1, 1, Math.max(2, Math.min(10000, maxDepth)));
  }

  /** {@inheritDoc} */
  @Override
  protected final String getJobName() {
    return GreedyMCCTreeTrainer.NAME;
  }

  /** the interval comparator */
  private static final class __IntervalComparator
      implements Comparator<double[]> {
    /** the index */
    private final int m_index;

    /**
     * create the interval comparator
     *
     * @param index
     *          the index
     */
    __IntervalComparator(final int index) {
      super();
      this.m_index = index;
    }

    /** {@inheritDoc} */
    @Override
    public final int compare(final double[] o1, final double[] o2) {
      if (o1 == o2) {
        return 0;
      }
      if (o1 == null) {
        return 1;
      }
      if (o2 == null) {
        return (-1);
      }
      return Double.compare(o1[this.m_index], o2[this.m_index]);
    }

  }
}
