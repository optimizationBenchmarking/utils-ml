package examples.org.optimizationBenchmarking.utils.ml.models;

import java.awt.Color;
import java.io.IOException;
import java.lang.ProcessBuilder.Redirect;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.functions.UnaryFunction;
import org.optimizationBenchmarking.utils.math.functions.power.Lg;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.processing.FunctionSamplingJob;
import org.optimizationBenchmarking.utils.math.text.ABCParameterRenderer;
import org.optimizationBenchmarking.utils.math.text.AbstractParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.numbers.SimpleNumberAppender;
import org.optimizationBenchmarking.utils.text.textOutput.AbstractTextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** color utilities */
abstract class _Utils {
  /** the minimal x coordinate */
  static final double MIN_X = 1d;
  /** the maximal x coordinate */
  static final double MAX_X = 1e5d;

  /** the shared instance */
  static final __XRenderer X_RENDERER = new __XRenderer();

  /**
   * plot a model to the specified target writer
   *
   * @param function
   *          the function to be plotted
   * @param target
   *          the target destination
   */
  public static final void _plotModel(final UnaryFunction function,
      final ITextOutput target) {
    final int size;
    final DoubleMatrix1D matrix;
    int index;

    target.appendLineBreak();
    target.appendLineBreak();
    target.appendLineBreak();
    target.append('#');
    target.append(' ');
    function.mathRender(target, _Utils.X_RENDERER);
    function.mathRender(AbstractTextOutput.wrap(System.out),
        _Utils.X_RENDERER);
    System.out.println();
    matrix = new FunctionSamplingJob(function, _Utils.MIN_X, _Utils.MAX_X,
        Lg.INSTANCE, 10000, -1).call();
    size = matrix.m();
    for (index = 0; index < size; index++) {
      target.appendLineBreak();
      _Utils.__render(matrix.getDouble(index, 0), target);
      target.append('\t');
      _Utils.__render(matrix.getDouble(index, 1), target);
    }
  }

  /**
   * create steps by linearly dividing the interval {@code [min,max]} into
   * {@code steps} values
   *
   * @param min
   *          the minimum value
   * @param max
   *          the maximum value
   * @param steps
   *          the number of steps
   * @return the interval
   */
  static final double[] _stepLinear(final double min, final double max,
      final int steps) {
    final double[] result;
    final double useMin, useMax;
    int index;

    result = new double[steps];
    useMin = Math.min(min, max);
    useMax = Math.max(min, max);
    for (index = steps; (--index) >= 0;) {
      result[index] = Math.max(useMin,
          Math.min(useMax, _Utils._smoothValue(
              (useMin + ((((useMax - useMin) * index) / (steps - 1)))))));
    }
    result[0] = useMin;
    result[result.length - 1] = useMax;
    Arrays.sort(result);
    return result;
  }

  /**
   * the original value
   *
   * @param value
   *          the value
   * @return the smoothed value
   */
  static final double _smoothValue(final double value) {
    double best, low, up;
    int bestQuality, currentQuality, index;

    up = low = best = value;
    bestQuality = _Utils.__quality(value);

    for (index = 10; (--index) > 0;) {
      up = Math.nextUp(best);
      currentQuality = _Utils.__quality(up);
      if (currentQuality < bestQuality) {
        best = up;
        bestQuality = currentQuality;
      }
      low = Math.nextAfter(low, Double.NEGATIVE_INFINITY);
      currentQuality = _Utils.__quality(low);
      if (currentQuality < bestQuality) {
        best = low;
        bestQuality = currentQuality;
      }
    }

    return best;
  }

  /**
   * get the quality
   *
   * @param value
   *          the value
   * @return the quality
   */
  private static final int __quality(final double value) {
    if (MathUtils.isFinite(value)) {
      return SimpleNumberAppender.INSTANCE
          .toString(value, ETextCase.IN_SENTENCE).length();
    }
    return Integer.MAX_VALUE;
  }

  /**
   * render a value
   *
   * @param value
   *          the value
   * @param textOut
   *          the output
   */
  private static final void __render(final double value,
      final ITextOutput textOut) {
    double use;

    if (value <= -1e300d) {
      use = -1e300d;
    } else {
      if (value >= 1e300d) {
        use = 1e300d;
      } else {
        if (value != value) {
          throw new IllegalStateException();
        }
        use = value;
      }
    }

    SimpleNumberAppender.INSTANCE.appendTo(use, ETextCase.IN_SENTENCE,
        textOut);

  }

  /**
   * get a good name for figures and files
   *
   * @param model
   *          the model
   * @param index
   *          the index
   * @return the name
   */
  static final String _functionFileName(
      final ParametricUnaryFunction model, final int index) {
    return Integer.toString(index + 1) + '_'
        + model.getClass().getSimpleName();
  }

  /**
   * blend the colors
   *
   * @param color
   *          the color
   * @return the color string
   */
  static final char[] _colorName(final Color color) {
    final char[] sb;

    sb = new char[13];
    sb[0] = 'r';
    sb[1] = 'g';
    sb[2] = 'b';
    sb[3] = ' ';
    sb[4] = '"';
    sb[5] = '#';

    sb[6] = Character.forDigit(color.getRed() / 16, 16);
    sb[7] = Character.forDigit(color.getRed() % 16, 16);
    sb[8] = Character.forDigit(color.getGreen() / 16, 16);
    sb[9] = Character.forDigit(color.getGreen() % 16, 16);
    sb[10] = Character.forDigit(color.getBlue() / 16, 16);
    sb[11] = Character.forDigit(color.getBlue() % 16, 16);
    sb[12] = '"';

    return sb;
  }

  /**
   * blend colors, taken from http://stackoverflow.com/questions/21270610
   *
   * @param colors
   *          the colors
   * @param fractions
   *          the color factions
   * @param progress
   *          the progress
   * @return the color
   */
  static final Color _colorBlend(final Color[] colors,
      final double[] fractions, final double progress) {

    final int[] indicies = _Utils.__colorGetFractionIndicies(fractions,
        progress);

    final double[] range = new double[] { fractions[indicies[0]],
        fractions[indicies[1]] };
    final Color[] colorRange = new Color[] { colors[indicies[0]],
        colors[indicies[1]] };

    final double max = range[1] - range[0];
    final double value = progress - range[0];
    final double weight = value / max;

    return _Utils._colorBlend(colorRange[0], colorRange[1], 1d - weight);
  }

  /**
   * get the fraction indices, taken from
   * http://stackoverflow.com/questions/21270610
   *
   * @param fractions
   *          the fractions
   * @param progress
   *          the progress
   * @return the indices
   */
  private static final int[] __colorGetFractionIndicies(
      final double[] fractions, final double progress) {
    final int[] range = new int[2];

    int startPoint = 0;
    while ((startPoint < fractions.length)
        && (fractions[startPoint] <= progress)) {
      startPoint++;
    }

    if (startPoint >= fractions.length) {
      startPoint = fractions.length - 1;
    }

    range[0] = startPoint - 1;
    range[1] = startPoint;

    return range;
  }

  /**
   * blend the colors, taken from
   * http://stackoverflow.com/questions/21270610
   *
   * @param color1
   *          the first color
   * @param color2
   *          the second color
   * @param ratio
   *          the ratio
   * @return the blended color
   */
  static final Color _colorBlend(final Color color1, final Color color2,
      final double ratio) {
    final double r = ratio;
    final double ir = 1d - r;

    final float rgb1[] = new float[3];
    final float rgb2[] = new float[3];

    color1.getColorComponents(rgb1);
    color2.getColorComponents(rgb2);

    float red = (float) ((rgb1[0] * r) + (rgb2[0] * ir));
    float green = (float) ((rgb1[1] * r) + (rgb2[1] * ir));
    float blue = (float) ((rgb1[2] * r) + (rgb2[2] * ir));

    if (red < 0) {
      red = 0;
    } else
      if (red > 1) {
        red = 1;
      }
    if (green < 0) {
      green = 0;
    } else
      if (green > 1) {
        green = 1;
      }
    if (blue < 0) {
      blue = 0;
    } else
      if (blue > 1) {
        blue = 1;
      }

    return new Color(red, green, blue);
  }

  /**
   * set up a gnuplot script
   *
   * @param vector
   *          create vector graphics ({@code true}) or pixel graphics (
   *          {@code false})
   * @param gnuplotScript
   *          the gnuplot script
   * @param useYRange
   *          should we set the y range
   */
  static final void _gnuplotSetup(final boolean vector,
      final ITextOutput gnuplotScript, final boolean useYRange) {
    gnuplotScript.append(vector//
        ? "set terminal postscript eps enhanced size 5,3"//$NON-NLS-1$
        : "set terminal png truecolor rounded size 1280,1024 crop enhanced");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    if (useYRange) {
      gnuplotScript.append("set yrange [0:100]");//$NON-NLS-1$
      gnuplotScript.appendLineBreak();
    }
    gnuplotScript.append("set xrange [0:5]");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset xtics");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset ytics");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset mxtics");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset mytics");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("set format x \"\"");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("set format y \"\"");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset key");//$NON-NLS-1$
    gnuplotScript.appendLineBreak();
    gnuplotScript.appendLineBreak();
  }

  /**
   * run gnuplot the gnutplot
   *
   * @param destDir
   *          the destination folder
   * @param gnuplotScriptPath
   *          the script path
   */
  static final void _gnuplotRun(final Path destDir,
      final Path gnuplotScriptPath) {
    try {
      new ProcessBuilder()//
          .command("gnuplot", //$NON-NLS-1$
              gnuplotScriptPath.toFile().getCanonicalPath())//
          .directory(destDir.toFile())//
          .redirectInput(Redirect.INHERIT)//
          .redirectError(Redirect.INHERIT)//
          .start().waitFor();
    } catch (final Throwable error) {
      error.printStackTrace();
    }
  }

  /**
   * set up a gnuplot graphic
   *
   * @param gnuplotScript
   *          the gnuplot script
   * @param outputFile
   *          the output eps file
   * @param model
   *          the model
   * @param parameters
   *          the parameters to render
   */
  static final void _gnuplotBeginPlot(final ITextOutput gnuplotScript,
      final String outputFile, final ParametricUnaryFunction model,
      final double[] parameters) {

    gnuplotScript.appendLineBreak();
    gnuplotScript.append("# model: "); //$NON-NLS-1$
    model.mathRender(gnuplotScript, ABCParameterRenderer.INSTANCE);
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("# base setup: "); //$NON-NLS-1$
    model.toUnaryFunction(parameters).mathRender(gnuplotScript,
        _Utils.X_RENDERER);
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("set output '"); //$NON-NLS-1$
    gnuplotScript.append(outputFile);
    gnuplotScript.append('\'');
  }

  /**
   * create the output file name
   *
   * @param name
   *          the base file name
   * @param vector
   *          create vector graphics ({@code true}) or pixel graphics (
   *          {@code false})
   * @return the output file name
   */
  static final String _gnuplotFigureFile(final String name,
      final boolean vector) {
    return name + (vector ? ".eps" : ".png"); //$NON-NLS-1$ //$NON-NLS-2$
  }

  /**
   * set up a gnuplot graphic
   *
   * @param gnuplotScript
   *          the gnuplot script
   * @param inputFile
   *          the input text file
   * @param indexOfRun
   *          the index of the run
   * @param title
   *          the title
   * @param color
   *          the color
   * @param lineWidth
   *          the line width
   */
  static final void _gnuplotCurve(final ITextOutput gnuplotScript,
      final String inputFile, final int indexOfRun, final String title,
      final Color color, final int lineWidth) {
    _Utils._gnuplotCurve(gnuplotScript, inputFile, indexOfRun,
        (indexOfRun <= 0), title, color, lineWidth);
  }

  /**
   * set up a gnuplot graphic
   *
   * @param gnuplotScript
   *          the gnuplot script
   * @param inputFile
   *          the input text file
   * @param indexOfRun
   *          the index of the run
   * @param isFirst
   *          is this the first line to plot?
   * @param title
   *          the title
   * @param color
   *          the color
   * @param lineWidth
   *          the line width
   */
  static final void _gnuplotCurve(final ITextOutput gnuplotScript,
      final String inputFile, final int indexOfRun, final boolean isFirst,
      final String title, final Color color, final int lineWidth) {

    if (!isFirst) {
      gnuplotScript.append(',');
      gnuplotScript.append('\\');
    }
    gnuplotScript.appendLineBreak();
    if (isFirst) {
      gnuplotScript.append("plot '"); //$NON-NLS-1$
    } else {
      gnuplotScript.append("     '"); //$NON-NLS-1$
    }
    gnuplotScript.append(inputFile);
    gnuplotScript.append("' index "); //$NON-NLS-1$
    gnuplotScript.append(indexOfRun);
    gnuplotScript.append(" title '"); //$NON-NLS-1$
    gnuplotScript.append(title);
    gnuplotScript.append("' smooth mcsplines lt 1 lc "); //$NON-NLS-1$
    gnuplotScript.append(_Utils._colorName(color));
    if (lineWidth > 0) {
      gnuplotScript.append(" lw "); //$NON-NLS-1$
      gnuplotScript.append(lineWidth);
    }
  }

  /**
   * tear down a gnuplot graphic
   *
   * @param gnuplotScript
   *          the gnuplot script
   */
  static final void _gnuplotEndPlot(final ITextOutput gnuplotScript) {
    gnuplotScript.appendLineBreak();
    gnuplotScript.append("unset output"); //$NON-NLS-1$
    gnuplotScript.appendLineBreak();
  }

  /**
   * get the destination path
   *
   * @param args
   *          the command line arguments provided to the program
   * @param clazz
   *          the calling class
   * @return the path
   * @throws IOException
   *           if i/o fails
   */
  static final Path _destination(final String[] args, final Class<?> clazz)
      throws IOException {
    Path path;

    if ((args != null) && (args.length > 0)) {
      path = Paths.get(args[0]);
    } else {
      path = PathUtils.createPathInside(PathUtils.getTempDir(),
          clazz.getSimpleName());
    }
    Files.createDirectories(path);
    return path;
  }

  /** render {@code x} */
  private static final class __XRenderer
      extends AbstractParameterRenderer {

    /** create */
    __XRenderer() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final void renderParameter(final int index,
        final ITextOutput out) {
      if (index == 0) {
        out.append('x');
      } else {
        AbstractParameterRenderer.throwInvalidParameterIndex(index, 0);
      }
    }

    /** {@inheritDoc} */
    @Override
    public final void renderParameter(final int index, final IMath out) {
      if (index == 0) {
        try (final IText text = out.name()) {
          text.append('x');
        }
      } else {
        AbstractParameterRenderer.throwInvalidParameterIndex(index, 0);
      }
    }
  }
}
