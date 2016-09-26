package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import java.util.Arrays;
import java.util.Comparator;

import org.optimizationBenchmarking.utils.math.combinatorics.CanonicalPermutation;
import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;

/** A set of classified samples */
public class ClassifiedSampleInfo {

  /** the classes which actually occured */
  private final int[] m_classes;

  /** the reverse-lookup for class indexes */
  private final int[] m_classIndexes;

  /** the sizes of the classes */
  private final int[] m_classSizes;

  /** the largest class */
  private final int m_largestIndex;

  /** the smallest class index */
  private final int m_smallestIndex;

  /** the total number of samples */
  private final int m_sampleSize;

  /**
   * create the classified sample set
   *
   * @param samples
   *          the samples
   */
  public ClassifiedSampleInfo(final ClassifiedSample... samples) {
    super();

    int[][] classes, copy;
    int i, size, maxClass, minClass;

    classes = new int[samples.length][];
    size = 0;
    maxClass = 0;
    samplesLoop: for (final ClassifiedSample sample : samples) {
      maxClass = Math.max(maxClass, sample.sampleClass);
      for (i = size; (--i) >= 0;) {
        if (sample.sampleClass == classes[i][0]) {
          ++classes[i][1];
          continue samplesLoop;
        }
      }
      classes[size++] = new int[] { sample.sampleClass, 1 };
    }

    if (size < classes.length) {
      copy = new int[size][];
      System.arraycopy(classes, 0, copy, 0, size);
      classes = copy;
    }
    Arrays.sort(classes, new _ClassesSorter());

    this.m_classes = new int[classes.length];
    this.m_classSizes = new int[classes.length];
    this.m_classIndexes = new int[maxClass + 1];

    i = 0;
    minClass = maxClass = 0;
    for (final int[] clazz : classes) {
      this.m_classes[i] = clazz[0];
      this.m_classIndexes[clazz[0]] = i;
      this.m_classSizes[i] = clazz[1];

      if (clazz[1] > this.m_classSizes[maxClass]) {
        maxClass = i;
      }
      if (clazz[1] < this.m_classSizes[minClass]) {
        minClass = i;
      }

      ++i;
    }

    this.m_largestIndex = maxClass;
    this.m_smallestIndex = minClass;
    this.m_sampleSize = samples.length;
  }

  /**
   * create the classified sample set
   *
   * @param sizes
   *          the class sizes
   */
  ClassifiedSampleInfo(final int[] sizes) {
    super();
    this.m_classes = this.m_classIndexes = CanonicalPermutation
        .createCanonicalZero(sizes.length);
    this.m_classSizes = sizes;

    int count, max, min, index;
    count = max = min = 0;
    index = (-1);
    for (final int size : sizes) {
      count += size;
      ++index;
      if (size > sizes[max]) {
        max = index;
      }
      if (size < sizes[min]) {
        min = index;
      }
    }
    this.m_sampleSize = count;
    this.m_largestIndex = max;
    this.m_smallestIndex = min;
  }

  /**
   * Get the total size of the sample
   *
   * @return the total size of the sample
   */
  public final int getSampleSize() {
    return this.m_sampleSize;
  }

  /**
   * Get the actual number of classes in the sample
   *
   * @return the actual number of classes in the sample
   */
  public final int getClassCount() {
    return this.m_classes.length;
  }

  /**
   * Get the index of the given sample class
   *
   * @param sampleClass
   *          the sample class
   * @return the sample class index
   */
  public final int getClassIndex(final int sampleClass) {
    return this.m_classIndexes[sampleClass];
  }

  /**
   * Get the sample class for a given class index
   *
   * @param classIndex
   *          the class index
   * @return the sample class belonging to that index
   */
  public final int getSampleClass(final int classIndex) {
    return this.m_classes[classIndex];
  }

  /**
   * Get the size of the given index class
   *
   * @param indexClass
   *          the index class
   * @return the size of that class
   */
  public final int getIndexClassSize(final int indexClass) {
    return this.m_classSizes[indexClass];
  }

  /**
   * Get the size of the given sample class
   *
   * @param sampleClass
   *          the sample class
   * @return the size of that class
   */
  public final int getSampleClassSize(final int sampleClass) {
    return this.m_classSizes[this.getClassIndex(sampleClass)];
  }

  /**
   * Get the index of the largest index class
   *
   * @return the index of the largest index class
   */
  public final int getBiggestIndexClass() {
    return this.m_largestIndex;
  }

  /**
   * Get the index of the largest sample class
   *
   * @return the index of the largest sample class
   */
  public final int getBiggestSampleClass() {
    return this.m_classes[this.getBiggestIndexClass()];
  }

  /**
   * Get the index of the smallest index class
   *
   * @return the index of the smallest index class
   */
  public final int getSmallestIndexClass() {
    return this.m_smallestIndex;
  }

  /**
   * Get the index of the smallest sample class
   *
   * @return the index of the smallest sample class
   */
  public final int getSmallestSampleClass() {
    return this.m_classes[this.getSmallestIndexClass()];
  }

  /**
   * Map a classified sample from sample to index class space
   *
   * @param orig
   *          the original sample
   * @return the mapped sample
   */
  public final ClassifiedSample map(final ClassifiedSample orig) {
    final int newClass;

    newClass = this.getClassIndex(orig.sampleClass);
    if (newClass == orig.sampleClass) {
      return orig;
    }
    return new ClassifiedSample(newClass, orig.featureValues);
  }

  /**
   * Map a classified sample from sample to index class space
   *
   * @param orig
   *          the original sample
   * @return the mapped sample
   */
  public final ClassifiedSample[] map(final ClassifiedSample... orig) {
    int newClass, index;
    ClassifiedSample[] samples;

    samples = orig;
    index = (-1);
    for (final ClassifiedSample sample : samples) {
      ++index;
      newClass = this.getClassIndex(sample.sampleClass);
      if (newClass == sample.sampleClass) {
        continue;
      }
      if (orig == samples) {
        samples = samples.clone();
      }
      samples[index] = new ClassifiedSample(newClass,
          sample.featureValues);
    }
    return samples;
  }

  /** sort the classes */
  private static final class _ClassesSorter implements Comparator<int[]> {
    /** create */
    _ClassesSorter() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final int compare(final int[] o1, final int[] o2) {
      int res;
      res = Integer.compare(o2[1], o1[1]);
      if (res == 0) {
        return Integer.compare(o1[0], o2[0]);
      }
      return res;
    }
  }

}
