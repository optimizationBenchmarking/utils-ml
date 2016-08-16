package examples.org.optimizationBenchmarking.utils.ml.models;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.lang.ProcessBuilder.Redirect;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.functions.UnaryFunction;
import org.optimizationBenchmarking.utils.math.functions.power.Lg;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.processing.FunctionSamplingJob;
import org.optimizationBenchmarking.utils.math.text.ABCParameterRenderer;
import org.optimizationBenchmarking.utils.math.text.AbstractParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExpLinearModelOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExponentialDecayModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.GompertzModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.LogisticModelWithOffsetOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.numbers.SimpleNumberAppender;
import org.optimizationBenchmarking.utils.text.textOutput.AbstractTextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** plot the model */
public class ModelPlotter {

  /** the minimal x coordinate */
  private static final double MIN_X = 1d;
  /** the maximal x coordinate */
  private static final double MAX_X = 1e5d;

  /**
   * plot a model to the specified target writer
   *
   * @param function
   *          the function to be plotted
   * @param target
   *          the target destination
   */
  public static final void plotModel(final UnaryFunction function,
      final ITextOutput target) {
    final DoubleMatrix1D matrix;
    final int size;
    int index;

    target.appendLineBreak();
    target.appendLineBreak();
    target.appendLineBreak();
    target.append('#');
    target.append(' ');
    function.mathRender(target, __XRenderer.INSTANCE);
    function.mathRender(AbstractTextOutput.wrap(System.out),
        __XRenderer.INSTANCE);
    System.out.println();

    matrix = new FunctionSamplingJob(function, ModelPlotter.MIN_X,
        ModelPlotter.MAX_X, Lg.INSTANCE, 10000, -1).call();
    size = matrix.m();
    for (index = 0; index < size; index++) {
      target.appendLineBreak();
      ModelPlotter.__render(matrix.getDouble(index, 0), target);
      target.append('\t');
      ModelPlotter.__render(matrix.getDouble(index, 1), target);
    }
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
   * plot a model for parameter step sizes
   *
   * @param model
   *          the model
   * @param script
   *          the gnuplot script to generate
   * @param name
   *          the model's name
   * @param parameters
   *          the parameters
   * @param stepParameterIndex
   *          the step parameter index
   * @param destDir
   *          the destination directory
   * @param parameterValues
   *          the values of the parameter to render
   * @throws IOException
   *           if i/o fails
   */
  public static final void plotModelStepWise(final Path destDir,
      final ITextOutput script, final ParametricUnaryFunction model,
      final String name, final double[] parameters,
      final int stepParameterIndex, final double... parameterValues)
          throws IOException {
    final Path path;
    final ITextOutput textOut;
    final double[] use;
    final String fileName, sourceFile;
    final char parameter;
    int index;

    parameter = (char) ('A' + stepParameterIndex);
    fileName = name + '_' + parameter;
    sourceFile = (fileName + ".txt"); //$NON-NLS-1$
    path = PathUtils.createPathInside(destDir, sourceFile);
    use = parameters.clone();

    script.append("# model "); //$NON-NLS-1$
    model.mathRender(script, ABCParameterRenderer.INSTANCE);

    script.appendLineBreak();
    script.appendLineBreak();
    script.append("set output '"); //$NON-NLS-1$
    script.append(fileName);
    script.append(".eps"); //$NON-NLS-1$

    try (final OutputStream outputStream = PathUtils
        .openOutputStream(path)) {
      try (final OutputStreamWriter outputWriter = new OutputStreamWriter(
          outputStream)) {
        textOut = AbstractTextOutput.wrap(outputWriter);
        textOut.append("# model "); //$NON-NLS-1$
        model.mathRender(textOut, ABCParameterRenderer.INSTANCE);
        textOut.appendLineBreak();
        textOut.append("# iterating parameter "); //$NON-NLS-1$
        textOut.append(stepParameterIndex);

        index = 0;
        for (final double value : parameterValues) {
          if (index > 0) {
            script.append(',');
            script.append('\\');
          }
          script.appendLineBreak();
          if (index == 0) {
            script.append("plot '"); //$NON-NLS-1$
          } else {
            script.append("     '"); //$NON-NLS-1$
          }
          script.append(sourceFile);
          script.append("' index "); //$NON-NLS-1$
          script.append(index);
          script.append(" title '"); //$NON-NLS-1$
          script.append(parameter);
          script.append('=');
          SimpleNumberAppender.INSTANCE.appendTo(value,
              ETextCase.IN_SENTENCE, script);
          script.append("' smooth mcsplines"); //$NON-NLS-1$

          use[stepParameterIndex] = value;
          ModelPlotter.plotModel(model.toUnaryFunction(use), textOut);
          index++;
        }
      }
    }
    script.appendLineBreak();
    script.append("unset output"); //$NON-NLS-1$
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_1(final Path destDir,
      final ITextOutput script) throws IOException {
    final ExponentialDecayModel model;
    final String name;
    double[] parameters, use;

    model = new ExponentialDecayModel();
    name = ((model.getClass().getSimpleName() + '_') + '1');

    parameters = new double[] { 0, 100, -0.01, 0.8d };

    use = parameters.clone();
    use[1] = 70;
    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        use, 0, //
        0, 15, 30, 45);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        55, 70, 85, 100);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        -0.09d, -0.06d, -0.03d, -0.01d);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        0.8d, 1.2d, 1.6d, 2d);
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_2(final Path destDir,
      final ITextOutput script) throws IOException {
    final ExponentialDecayModel model;
    final String name;
    double[] parameters, use;

    model = new ExponentialDecayModel();
    name = ((model.getClass().getSimpleName() + '_') + '2');

    parameters = new double[] { 100, -100, 1d / -0.01, -0.8d };

    use = parameters.clone();
    use[1] = -70;
    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        use, 0, //
        115, 100, 85, 70);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        -100, -85, -70, -55);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        1d / -0.09d, 1d / -0.06d, 1d / -0.03d, 1d / -0.01d);
    // -8d, -6d, -4d, -2d);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        -0.8d, -1.2d, -1.6d, -2d);
  }

  /**
   * plot an logistic model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_LogisticModelWithOffsetOverLogX(
      final Path destDir, final ITextOutput script) throws IOException {
    final LogisticModelWithOffsetOverLogX model;
    final String name;
    double[] parameters, use;

    model = new LogisticModelWithOffsetOverLogX();
    name = model.getClass().getSimpleName();

    parameters = new double[] { 100, -100, 1d / -0.01, -0.8d };

    use = parameters.clone();
    use[1] = -70;
    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        use, 0, //
        115, 100, 85, 70);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        -100, -85, -70, -55);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        1d / -0.09d, 1d / -0.06d, 1d / -0.03d, 1d / -0.01d);
    // -8d, -6d, -4d, -2d);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        -0.8d, -1.2d, -1.6d, -2d);
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_1(final Path destDir,
      final ITextOutput script) throws IOException {
    final GompertzModel model;
    final String name;
    double[] parameters, use;

    model = new GompertzModel();
    name = (model.getClass().getSimpleName() + '_') + '1';

    parameters = new double[] { 100, -100, -100, -0.05d };

    use = parameters.clone();
    use[1] = -70;
    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        use, 0, //
        115, 100, 85, 70);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        -100, -85, -70, -55);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        -12.5d, -2.5d, -0.5d, -0.1d);
    // -8d, -6d, -4d, -2d);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        -3.2, -0.8, -0.2, -0.050d);
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_2(final Path destDir,
      final ITextOutput script) throws IOException {
    final GompertzModel model;
    final String name;
    double[] parameters;

    model = new GompertzModel();
    name = (model.getClass().getSimpleName() + '_') + '2';

    parameters = new double[] { 0, 100, -0.1, 0.01d };

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 0, //
        0, 15, 30, 45);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        55, 70, 85, 100);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        -3.1, -2.1, -2.1, -0.1);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        0.01, 0.04, 0.07, 0.11);
  }

  /**
   * plot an exp-linear model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param script
   *          the gnuplot script to generate
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExpLinearModelOverLogX(final Path destDir,
      final ITextOutput script) throws IOException {
    final ExpLinearModelOverLogX model;
    final String name;
    double[] parameters, use;

    model = new ExpLinearModelOverLogX();
    name = model.getClass().getSimpleName();

    parameters = new double[] { 0, 100, -0.75d, 0 };

    use = parameters.clone();
    use[1] = 70;
    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        use, 0, //
        0, 15, 30, 45);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 1, //
        55, 70, 85, 100);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 2, //
        -1d, -0.75d, -0.5d, -0.25d);

    ModelPlotter.plotModelStepWise(destDir, script, model, name, //
        parameters, 3, //
        0, 3, 6, 9);
  }

  /**
   * the main routine
   *
   * @param args
   *          the command line arguments
   * @throws IOException
   *           if i/o fails
   * @throws InterruptedException
   *           if the gnuplot does not run as expected
   */
  public static final void main(final String[] args)
      throws IOException, InterruptedException {
    final Path dest, scriptPath;
    ITextOutput script;

    dest = Paths.get(args[0]);
    scriptPath = dest.resolve("gnuplot.script");//$NON-NLS-1$
    try (final BufferedWriter writer = Files.newBufferedWriter(scriptPath,
        Charset.forName("UTF-8"))) {//$NON-NLS-1$
      script = AbstractTextOutput.wrap(writer);

      script.append("set terminal eps size 5,3");//$NON-NLS-1$
      script.appendLineBreak();
      script.append("set yrange [0:100]");//$NON-NLS-1$
      script.appendLineBreak();
      script.append("set xrange [0:5]");//$NON-NLS-1$
      script.appendLineBreak();
      script.appendLineBreak();

      ModelPlotter.plot_ExponentialDecay_1(dest, script);
      ModelPlotter.plot_ExponentialDecay_2(dest, script);
      ModelPlotter.plot_LogisticModelWithOffsetOverLogX(dest, script);
      ModelPlotter.plot_GompertzModel_1(dest, script);
      ModelPlotter.plot_GompertzModel_2(dest, script);
      ModelPlotter.plot_ExpLinearModelOverLogX(dest, script);
    }

    new ProcessBuilder()//
        .command("gnuplot", //$NON-NLS-1$
            scriptPath.toFile().getCanonicalPath())//
        .directory(dest.toFile())//
        .redirectErrorStream(true)//
        .redirectInput(Redirect.INHERIT)//
        .start().waitFor();
  }

  /** render {@code x} */
  private static final class __XRenderer
      extends AbstractParameterRenderer {

    /** the shared instance */
    static final __XRenderer INSTANCE = new __XRenderer();

    /** create */
    private __XRenderer() {
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
