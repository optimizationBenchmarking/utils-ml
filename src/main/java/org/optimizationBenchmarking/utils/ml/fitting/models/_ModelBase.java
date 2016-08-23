package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.functions.arithmetic.AddN;

/** A base class with some utility methods */
abstract class _ModelBase extends BasicModel {

  /** create */
  _ModelBase() {
    super();
  }

  /**
   * compute the square
   *
   * @param a
   *          the number
   * @return the result
   */
  static final double _sqr(final double a) {
    return a * a;
  }

  /**
   * compute the power
   *
   * @param a
   *          the base
   * @param b
   *          the power
   * @return the result
   */
  static final double _pow(final double a, final double b) {
    final double res;
    res = Math.pow(a, b);
    return ((res != 0d) ? res : 0d);
  }

  /**
   * compute {@code exp(o*p)} and protect against NaN
   *
   * @param o
   *          the first number
   * @param p
   *          the second number
   * @return the result
   */
  static final double _exp_o_p(final double o, final double p) {
    final double res;
    if ((o == 0d) || (p == 0d)) {
      return 1d; // guard against infinity*0
    }
    res = Math.exp(o * p);
    return ((res == res) ? res : 0d);// guard against NaN
  }

  /**
   * sum up some numbers
   *
   * @param summands
   *          the numbers to sum up
   * @return the sum
   */
  static final double _add(final double... summands) {
    return AddN.destructiveSum(summands);
  }

  /**
   * Format a gradient
   *
   * @param gradient
   *          the gradient to format
   * @param original
   *          the original parameter value used to determine gradients in
   *          dodgy situations
   * @return the formatted gradient value
   */
  static final double _gradient(final double gradient,
      final double original) {
    final double res;
    if ((gradient != gradient) || (gradient == 0d)) {
      return 0d;// return positive 0 on NaN, 0, and -0
    }
    if (gradient <= Double.NEGATIVE_INFINITY) {
      res = Math.ulp(original);
      return ((res < Double.POSITIVE_INFINITY) ? (-res) : (-1d));
    }
    if (gradient >= Double.POSITIVE_INFINITY) {
      res = Math.ulp(original);
      return ((res < Double.POSITIVE_INFINITY) ? res : 1d);
    }
    return gradient;
  }

  /**
   * Obtain a slightly randomized version of the minimal and maximal
   * {@code y}-coordinates
   *
   * @param useY
   *          {@code true} for using {@code y} coordinates, {@code false}
   *          for using {@code x} coordinates
   * @param min
   *          the real minimal {@code y}
   * @param max
   *          the real maximal {@code y}
   * @param points
   *          the points, or {@code null} if none are given
   * @param random
   *          the random number generator
   * @return the randomized
   */
  static final double[] _getMinMax(final boolean useY, final double min,
      final double max, final double[] points, final Random random) {
    int trials;
    double current, chosenMin, chosenMax, createdMin, createdMax;
    int i;

    // find minimum y
    findMin: {
      if ((points != null) && (random.nextInt(10) > 0)) {
        chosenMin = Double.POSITIVE_INFINITY;
        for (i = (points.length - (useY ? 1 : 2)); i >= 0; i -= 2) {
          current = points[i];
          if ((current < chosenMin) && MathUtils.isFinite(current)) {
            chosenMin = current;
          }
        }
        if (MathUtils.isFinite(chosenMin)) {
          break findMin;
        }
      }

      chosenMin = min;
      if (MathUtils.isFinite(chosenMin)) {
        break findMin;
      }
      chosenMin = ((chosenMin <= 0d) ? -1e3d
          : ((chosenMin >= 0d) ? 1e3d : 0d));
    }

    // find maximum y
    findMax: {
      if ((points != null) && (random.nextInt(10) > 0)) {
        chosenMax = Double.NEGATIVE_INFINITY;
        for (i = (points.length - (useY ? 1 : 2)); i >= 0; i -= 2) {
          current = points[i];
          if ((current > chosenMax) && MathUtils.isFinite(current)) {
            chosenMax = current;
          }
        }
        if (MathUtils.isFinite(chosenMax)) {
          break findMax;
        }
      }

      chosenMax = max;
      if (MathUtils.isFinite(chosenMax)) {
        break findMax;
      }
      chosenMax = ((chosenMax <= 0d) ? -1e3d
          : ((chosenMax >= 0d) ? 1e3d : 0d));
    }

    // fix strange effects
    if (chosenMax <= chosenMin) {
      current = ((chosenMin > 0d) ? (1.1d * chosenMin)
          : ((chosenMax < 0d) ? (0.9d * chosenMax) : 1d));
      if ((current > chosenMin) && MathUtils.isFinite(current)) {
        chosenMax = current;
      } else {
        current = ((chosenMax > 0d) ? (0.9d * chosenMax)
            : ((chosenMin < 0d) ? (1.1d * chosenMax) : 1d));
        if ((current < chosenMax) && MathUtils.isFinite(current)) {
          chosenMin = current;
        } else {
          current = Math.nextUp(chosenMin);
          if (MathUtils.isFinite(current)) {
            chosenMax = current;
          } else {
            current = Math.nextAfter(chosenMax, Double.NEGATIVE_INFINITY);
            if (MathUtils.isFinite(current)) {
              chosenMin = current;
            }
          }
        }
      }
    }

    // modify the boundaries
    for (trials = 100; (--trials) >= 0;) {

      createdMin = (0.05d * random.nextGaussian());
      if (chosenMin > 0d) {
        createdMin = (0.99d + createdMin);
      } else {
        createdMin = (1.01d + createdMin);
      }
      createdMin = (chosenMin * createdMin);

      createdMax = (0.05d * random.nextGaussian());
      if (chosenMax > 0d) {
        createdMax = (1.01d + createdMax);
      } else {
        createdMax = (0.99d + createdMax);
      }
      createdMax = (chosenMax * createdMax);

      if ((createdMin < createdMax) && MathUtils.isFinite(createdMin)
          && MathUtils.isFinite(createdMax)) {
        return new double[] { createdMin, createdMax };
      }
    }
    return new double[] { chosenMin, chosenMax };
  }
}
