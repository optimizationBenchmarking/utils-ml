package examples.org.optimizationBenchmarking.utils.ml.models;

import java.awt.Color;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.text.ABCParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExpLinearModelOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExponentialDecayModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.GompertzModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.LogisticModelWithOffsetOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.numbers.SimpleNumberAppender;
import org.optimizationBenchmarking.utils.text.textOutput.AbstractTextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.MemoryTextOutput;

/** plot the model */
public class ModelPlotter extends _Utils {

  /** the colors used for blending */
  private static final Color[] BLEND_COLORS = { Color.BLUE, Color.GREEN,
      Color.RED };

  /** the blend color fractions */
  private static final double[] BLEND_COLOR_FRACTIONS = { 0d, 0.5d, 1d };

  /** the default number of steps */
  private static final int STEPS = 41;

  /**
   * plot a model for parameter step sizes
   *
   * @param model
   *          the model
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
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
   * @return the file name
   * @throws IOException
   *           if i/o fails
   */
  public static final String plotModelStepWise(final Path destDir,
      final ITextOutput gnuplotScript, final ParametricUnaryFunction model,
      final String name, final double[] parameters,
      final int stepParameterIndex, final double... parameterValues)
          throws IOException {
    final Path path;
    final ITextOutput textOut;
    final double[] use;
    final String fileName, sourceFile, outFile;
    final char parameter;
    int index;

    parameter = ((char) ('A' + stepParameterIndex));
    fileName = name + '_' + parameter;
    sourceFile = (fileName + ".txt"); //$NON-NLS-1$
    path = PathUtils.createPathInside(destDir, sourceFile);
    use = parameters.clone();
    outFile = (fileName + ".eps"); //$NON-NLS-1$

    _Utils._gnuplotBeginPlot(gnuplotScript, outFile, model, parameters);
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
          _Utils._gnuplotCurve(gnuplotScript, sourceFile, index, //
              parameter + "="//$NON-NLS-1$
                  + SimpleNumberAppender.INSTANCE.toString(value,
                      ETextCase.IN_SENTENCE), //
              ModelPlotter.__blendColors(index, parameterValues.length), //
              -1);

          use[stepParameterIndex] = value;
          _Utils._plotModel(model.toUnaryFunction(use), textOut);
          index++;
        }
      }
    }
    _Utils._gnuplotEndPlot(gnuplotScript);
    return outFile;
  }

  /**
   * create steps by linearly dividing the interval {@code [min,max]} into
   * {@value #STEPS} values
   *
   * @param min
   *          the minimum value
   * @param max
   *          the maximum value
   * @return the interval
   */
  private static final double[] __stepLinear(final double min,
      final double max) {
    return _Utils._stepLinear(min, max, ModelPlotter.STEPS);
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_1(final Path destDir,
      final ITextOutput gnuplotScript, final ITextOutput latexScript,
      final int index) throws IOException {
    final ExponentialDecayModel model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new ExponentialDecayModel();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 0, 100, -0.01, 0.8d };

    use = parameters.clone();
    use[1] = 70;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(0, 50));

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(50, 100));

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(-0.1d, -0.01d));

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(0.3d, 3d));

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_2(final Path destDir,
      final ITextOutput gnuplotScript, final ITextOutput latexScript,
      final int index) throws IOException {
    final ExponentialDecayModel model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new ExponentialDecayModel();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 100, -100, 1d / -0.01, -0.8d };

    use = parameters.clone();
    use[1] = -70;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(70, 120));

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(-110, -40));

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(1d / -0.01d, 1d / -0.1d));
    // -8d, -6d, -4d, -2d);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(-0.3d, -3d));

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an logistic model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_LogisticModelWithOffsetOverLogX_1(
      final Path destDir, final ITextOutput gnuplotScript,
      final ITextOutput latexScript, final int index) throws IOException {
    final LogisticModelWithOffsetOverLogX model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new LogisticModelWithOffsetOverLogX();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 0, 100, 1e-4, 2 };

    use = parameters.clone();
    use[0] = 0;
    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 1, //
        B = ModelPlotter.__stepLinear(50, 200));

    use = parameters.clone();
    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 2, //
        C = ModelPlotter.__stepLinear(1e-5, 1e-3d));
    // 1e-6, 1e-4, 1e-2, 1, 2);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(0.5, 8));
    // 1, 3, 5, 7);
    use = parameters.clone();
    use[1] = 50;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(0, 50));
    // 0, 15, 30, 45);

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an logistic model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_LogisticModelWithOffsetOverLogX_2(
      final Path destDir, final ITextOutput gnuplotScript,
      final ITextOutput latexScript, final int index) throws IOException {
    final LogisticModelWithOffsetOverLogX model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new LogisticModelWithOffsetOverLogX();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 100, -100, 20, -0.5 };

    use = parameters.clone();
    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 1, //
        B = ModelPlotter.__stepLinear(-50, -200));

    use = parameters.clone();
    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 2, //
        C = ModelPlotter.__stepLinear(0.5, 30));
    // 1e-6, 1e-4, 1e-2, 1, 2);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(-2, -0.1));
    // 1, 3, 5, 7);
    use = parameters.clone();
    use[1] = -50;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(50, 100));
    // 0, 15, 30, 45);

    ModelPlotter.__doPlot(latexScript, index, model, use,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_1(final Path destDir,
      final ITextOutput gnuplotScript, final ITextOutput latexScript,
      final int index) throws IOException {
    final GompertzModel model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new GompertzModel();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 100, -100, -5, -0.03d };

    use = parameters.clone();
    use[1] = -50;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(50, 120));// 115, 100, 85, 70);

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(-120, -50));// -100, -85, -70, -55);

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(-12.5d, -0.1d));// -12.5d, -2.5d,
                                                      // -0.5d, -0.1d);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(-3.2d, -0.02d));// -3.2, -0.8, -0.2,
                                                      // -0.050d);

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_2(final Path destDir,
      final ITextOutput gnuplotScript, final ITextOutput latexScript,
      final int index) throws IOException {
    final GompertzModel model;
    final String name, nA, nB, nC, nD;
    double[] parameters, A, B, C, D;

    model = new GompertzModel();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 0, 100, -0.1, 0.01d };

    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 0, //
        A = ModelPlotter.__stepLinear(0, 50));// 0, 15, 30, 45);

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(50, 100));// 55, 70, 85, 100);

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(-3.2d, -0.05d));// -3.1, -2.1, -2.1,
                                                      // -0.1);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(0.005, 0.15));// 0.01, 0.04, 0.07,
                                                    // 0.11);

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an exp-linear model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExpLinearModelOverLogX_1(
      final Path destDir, final ITextOutput gnuplotScript,
      final ITextOutput latexScript, final int index) throws IOException {
    final ExpLinearModelOverLogX model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new ExpLinearModelOverLogX();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 0, 175, -0.4d, 4 };

    use = parameters.clone();
    use[1] = 70;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(0, 50));// 0, 15, 30, 45);

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(50, 200));// 55, 70, 85, 100);

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(-1, -0.3));// -1d, -0.75d, -0.5d,
                                                 // -0.25d);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(0, 10));// 0, 3, 6, 9);

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
  }

  /**
   * plot an exp-linear model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param latexScript
   *          the latex script
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExpLinearModelOverLogX_2(
      final Path destDir, final ITextOutput gnuplotScript,
      final ITextOutput latexScript, final int index) throws IOException {
    final ExpLinearModelOverLogX model;
    final String name, nA, nB, nC, nD;
    double[] parameters, use, A, B, C, D;

    model = new ExpLinearModelOverLogX();
    name = _Utils._functionFileName(model, index);

    parameters = new double[] { 100, -1, 0.4d, 4 };

    use = parameters.clone();
    use[1] = -5;
    nA = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        use, 0, //
        A = ModelPlotter.__stepLinear(50, 120));// 0, 15, 30, 45);

    nB = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 1, //
        B = ModelPlotter.__stepLinear(-25, -0.05));// 55, 70, 85, 100);

    nC = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 2, //
        C = ModelPlotter.__stepLinear(1, 0.3));// -1d, -0.75d, -0.5d,
                                               // -0.25d);

    nD = ModelPlotter.plotModelStepWise(destDir, gnuplotScript, model,
        name, //
        parameters, 3, //
        D = ModelPlotter.__stepLinear(-0.9, 1e4));// 0, 3, 6, 9);

    ModelPlotter.__doPlot(latexScript, index, model, parameters,
        new String[] { nA, nB, nC, nD }, new double[][] { A, B, C, D });
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
    final Path dest, gnuplotScriptPath, latexScriptPath;
    ITextOutput gnuplotScript, latexScript;
    int index;

    dest = Paths.get(args[0]);
    gnuplotScriptPath = dest.resolve("gnuplot.gnuplotScript");//$NON-NLS-1$
    latexScriptPath = dest.resolve("models.tex");//$NON-NLS-1$
    try (final BufferedWriter gnuplotWriter = Files
        .newBufferedWriter(gnuplotScriptPath, Charset.forName("UTF-8"))) {//$NON-NLS-1$
      try (final BufferedWriter laTeXWriter = Files
          .newBufferedWriter(latexScriptPath, Charset.forName("UTF-8"))) {//$NON-NLS-1$

        latexScript = AbstractTextOutput.wrap(laTeXWriter);

        latexScript.append("\\begin{figure}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\definecolor{modelChartMinColor}{RGB}{");//$NON-NLS-1$
        latexScript.append(ModelPlotter.BLEND_COLORS[0].getRed());
        latexScript.append(',');
        latexScript.append(ModelPlotter.BLEND_COLORS[0].getGreen());
        latexScript.append(',');
        latexScript.append(ModelPlotter.BLEND_COLORS[0].getBlue());
        latexScript.append("}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\definecolor{modelChartMaxColor}{RGB}{");//$NON-NLS-1$
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length - 1]
                .getRed());
        latexScript.append(',');
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length - 1]
                .getGreen());
        latexScript.append(',');
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length - 1]
                .getBlue());
        latexScript.append("}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\definecolor{modelChartMidColor}{RGB}{");//$NON-NLS-1$
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length >>> 1]
                .getRed());
        latexScript.append(',');
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length >>> 1]
                .getGreen());
        latexScript.append(',');
        latexScript.append(
            ModelPlotter.BLEND_COLORS[ModelPlotter.BLEND_COLORS.length >>> 1]
                .getBlue());
        latexScript.append("}%");//$NON-NLS-1$
        latexScript.appendLineBreak();

        latexScript.append("\\resizebox{\\linewidth}{!}{%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\parbox{1.25\\linewidth}{%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\scriptsize%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\setstretch{1.0}%");//$NON-NLS-1$
        latexScript.appendLineBreak();

        gnuplotScript = AbstractTextOutput.wrap(gnuplotWriter);
        _Utils._gnuplotSetup(gnuplotScript, true);

        index = 0;
        ModelPlotter.plot_ExponentialDecay_1(dest, gnuplotScript,
            latexScript, index++);
        ModelPlotter.plot_ExponentialDecay_2(dest, gnuplotScript,
            latexScript, index++);
        ModelPlotter.plot_LogisticModelWithOffsetOverLogX_1(dest,
            gnuplotScript, latexScript, index++);
        ModelPlotter.plot_LogisticModelWithOffsetOverLogX_2(dest,
            gnuplotScript, latexScript, index++);
        ModelPlotter.plot_GompertzModel_1(dest, gnuplotScript, latexScript,
            index++);
        ModelPlotter.plot_GompertzModel_2(dest, gnuplotScript, latexScript,
            index++);
        ModelPlotter.plot_ExpLinearModelOverLogX_1(dest, gnuplotScript,
            latexScript, index++);
        ModelPlotter.plot_ExpLinearModelOverLogX_2(dest, gnuplotScript,
            latexScript, index++);

        latexScript.append('%');
        latexScript.appendLineBreak();
        latexScript.append("}}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append('%');
        latexScript.appendLineBreak();
        latexScript.append("\\vspace{-0.6em}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\caption{");//$NON-NLS-1$
        latexScript.append(
            "Different models and examples for the parameterizations over a log-scaled $x$ axis with $x\\in[1,1E5]$ and $y\\in[0,100]$. ");//$NON-NLS-1$
        latexScript.append(
            "Each of the four figures per row lets one of the parameters $A$, $B$, $C$, or $D$ (left to right) ");//$NON-NLS-1$
        latexScript.append(
            "range from a \\textcolor{modelChartMinColor}{smaller} over a \\textcolor{modelChartMidColor}{mid} to a \\textcolor{modelChartMaxColor}{larger} value, in ");//$NON-NLS-1$
        latexScript.append(ModelPlotter.STEPS);
        latexScript.append(
            " steps of equal length, while keeping the other parameters constant.");//$NON-NLS-1$
        latexScript.append("}%");//$NON-NLS-1$
        latexScript.append("\\label{fig:modelChart}%");//$NON-NLS-1$
        latexScript.appendLineBreak();
        latexScript.append("\\end{figure}%");//$NON-NLS-1$
        latexScript.appendLineBreak();

      }
    }

    _Utils._gnuplotRun(dest, gnuplotScriptPath);
  }

  /**
   * blend the colors
   *
   * @param index
   *          the index
   * @param length
   *          the length
   * @return the color
   */
  private static final Color __blendColors(final int index,
      final int length) {
    return _Utils._colorBlend(ModelPlotter.BLEND_COLORS,
        ModelPlotter.BLEND_COLOR_FRACTIONS,
        index / (ModelPlotter.STEPS - 1d));
  }

  /**
   * Print a plot.
   *
   * @param dest
   *          the destination text output
   * @param index
   *          the index
   * @param model
   *          the model
   * @param baseParams
   *          the basic parameters
   * @param files
   *          the files
   * @param parameterization
   *          the parameterizations
   */
  private static final void __doPlot(final ITextOutput dest,
      final int index, final ParametricUnaryFunction model,
      final double[] baseParams, final String[] files,
      final double[][] parameterization) {
    int xindex;
    MemoryTextOutput mto;
    String name;

    dest.append('%');
    dest.appendLineBreak();
    if (index > 0) {
      dest.append("\\\\");//$NON-NLS-1$
    }
    dest.append("\\noindent\\textbf{");//$NON-NLS-1$
    dest.append((char) ('a' + index));
    dest.append(".}~"); //$NON-NLS-1$
    mto = new MemoryTextOutput();
    model.printLongName(mto, ETextCase.AT_SENTENCE_START);
    name = mto.toString();
    model.printLongName(dest, ETextCase.AT_SENTENCE_START);
    if (!(name.toLowerCase().contains("model"))) { //$NON-NLS-1$
      dest.append(" model"); //$NON-NLS-1$
    }
    dest.append(" \\"); //$NON-NLS-1$
    dest.append(model.getClass().getSimpleName());

    for (xindex = 0; xindex < files.length; xindex++) {
      dest.append('{');
      dest.append((char) ('A' + xindex));
      dest.append('}');
    }
    dest.append("{x} for "); //$NON-NLS-1$

    for (xindex = 0; xindex < files.length; xindex++) {
      if (xindex > 0) {
        dest.append(", "); //$NON-NLS-1$
        if (xindex >= (files.length - 1)) {
          dest.append("and "); //$NON-NLS-1$
        }
      }
      dest.append("\\mbox{\\ensuremath{"); //$NON-NLS-1$
      dest.append((char) ('A' + xindex));
      dest.append(
          "\\in\\left[\\textnormal{\\textcolor{modelChartMinColor}{"); //$NON-NLS-1$
      SimpleNumberAppender.INSTANCE
          .appendTo(
              Math.min(Math.min(parameterization[xindex][0],
                  parameterization[xindex][parameterization[xindex].length
                      - 1]),
                  baseParams[xindex]),
              ETextCase.IN_SENTENCE, dest);
      dest.append("}},\\textnormal{\\textcolor{modelChartMaxColor}{"); //$NON-NLS-1$
      SimpleNumberAppender.INSTANCE
          .appendTo(
              Math.max(Math.max(parameterization[xindex][0],
                  parameterization[xindex][parameterization[xindex].length
                      - 1]),
                  baseParams[xindex]),
              ETextCase.IN_SENTENCE, dest);
      dest.append("}}\\right]}}"); //$NON-NLS-1$
    }
    dest.append(".\\\\%"); //$NON-NLS-1$
    dest.appendLineBreak();

    for (xindex = 0; xindex < files.length; xindex++) {
      if (xindex <= 0) {
        dest.append("\\noindent"); //$NON-NLS-1$
      } else {
        dest.append("\\strut\\hfill"); //$NON-NLS-1$
      }
      dest.append("\\strut%"); //$NON-NLS-1$
      dest.appendLineBreak();
      dest.append(
          "\\includegraphics[width=0.2475\\linewidth]{\\modelsPath/");//$NON-NLS-1$
      dest.append(files[xindex]);
      dest.append("}%"); //$NON-NLS-1$
      dest.appendLineBreak();
    }
    dest.append("\\strut%"); //$NON-NLS-1$
    dest.appendLineBreak();
    dest.append('%');
    dest.appendLineBreak();
  }
}
