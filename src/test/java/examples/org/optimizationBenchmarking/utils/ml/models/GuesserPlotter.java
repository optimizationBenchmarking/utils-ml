package examples.org.optimizationBenchmarking.utils.ml.models;

import java.awt.Color;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.functions.UnaryFunction;
import org.optimizationBenchmarking.utils.math.matrix.processing.FunctionSamplingJob;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExpLinearModelOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExponentialDecayModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.GompertzModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.LogisticModelWithOffsetOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.textOutput.AbstractTextOutput;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** plot the model */
public class GuesserPlotter extends _Utils {
  /** the default number of steps */
  private static final int STEPS = 500;

  /**
   * get the parameters
   *
   * @param parameters
   *          the parameters
   * @param a
   *          the first color
   * @param b
   *          the second color
   * @return the return value
   */
  private static final Color __color(final double[] parameters,
      final Color a, final Color b) {
    return _Utils._colorBlend(//
        _Utils._colorBlend(
            ((parameters[0] < 0d) ? a.brighter() : b.brighter()), //
            ((parameters[1] < 0d) ? a : b), //
            0.5d), //
        _Utils._colorBlend(
            ((parameters[2] < 0d) ? a.darker() : b.darker()), //
            ((parameters[3] < 0d) ? a.darker().darker()
                : b.darker().darker()), //
            0.5d), //
        0.5d);
  }

  /**
   * get the parameters
   *
   * @param parameters
   *          the parameters
   * @return the return value
   */
  private static final Color __color1(final double[] parameters) {
    return GuesserPlotter.__color(parameters, Color.red, Color.blue);
  }

  /**
   * get the parameters
   *
   * @param parameters
   *          the parameters
   * @return the return value
   */
  private static final Color __color2(final double[] parameters) {
    return GuesserPlotter.__color(parameters, Color.cyan, Color.orange);
  }

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
   * @param destDir
   *          the destination directory
   * @return the file name
   * @throws IOException
   *           if i/o fails
   */
  public static final String plotModelGuesses(final Path destDir,
      final ITextOutput gnuplotScript, final ParametricUnaryFunction model,
      final String name, final double... parameters) throws IOException {
    final Path path;
    final ITextOutput textOut;
    final String sourceFile, outFile;
    final IParameterGuesser guesser;
    final UnaryFunction function;
    final Random random;
    double[] guess;
    int index;

    sourceFile = (name + ".txt"); //$NON-NLS-1$
    path = PathUtils.createPathInside(destDir, sourceFile);

    outFile = _Utils._gnuplotFigureFile(name, false);

    function = model.toUnaryFunction(parameters);
    guesser = model.createParameterGuesser(
        new FunctionSamplingJob(function, _Utils.MIN_X, _Utils.MAX_X)
            .call());
    random = new Random();

    _Utils._gnuplotBeginPlot(gnuplotScript, outFile, model, parameters);
    try (final OutputStream outputStream = PathUtils
        .openOutputStream(path)) {
      try (final OutputStreamWriter outputWriter = new OutputStreamWriter(
          outputStream)) {
        textOut = AbstractTextOutput.wrap(outputWriter);
        _Utils._plotModel(model.toUnaryFunction(parameters), textOut);

        for (index = 1; index < GuesserPlotter.STEPS; index++) {
          guess = new double[parameters.length];
          guesser.createRandomGuess(guess, random);
          _Utils._gnuplotCurve(gnuplotScript, sourceFile, index,
              (index <= 1), //
              Arrays.toString(guess),

              (index < (GuesserPlotter.STEPS - 10))
                  ? GuesserPlotter.__color1(guess)
                  : GuesserPlotter.__color2(guess),

              (index < (GuesserPlotter.STEPS - 10)) ? -1 : 5);
          _Utils._plotModel(model.toUnaryFunction(guess), textOut);
        }
      }
    }

    _Utils._gnuplotCurve(gnuplotScript, sourceFile, 0, false, //
        Arrays.toString(parameters), Color.GREEN, 10);

    _Utils._gnuplotEndPlot(gnuplotScript);
    return outFile;
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_1(final Path destDir,
      final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final ExponentialDecayModel model;
    final String name;

    model = new ExponentialDecayModel();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        1, 98, -0.01, 0.8d);
  }

  /**
   * plot an exponential decay model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay_2(final Path destDir,
      final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final ExponentialDecayModel model;
    final String name;

    model = new ExponentialDecayModel();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        100, -100, 1d / -0.01, -0.8d);
  }

  /**
   * plot an logistic model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_LogisticModelWithOffsetOverLogX_1(
      final Path destDir, final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final LogisticModelWithOffsetOverLogX model;
    final String name;

    model = new LogisticModelWithOffsetOverLogX();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        0, 100, 1e-4, 2);
  }

  /**
   * plot an logistic model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_LogisticModelWithOffsetOverLogX_2(
      final Path destDir, final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final LogisticModelWithOffsetOverLogX model;
    final String name;

    model = new LogisticModelWithOffsetOverLogX();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        100, -100, 20, -0.5);
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_1(final Path destDir,
      final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final GompertzModel model;
    final String name;

    model = new GompertzModel();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        100, -100, -5, -0.03d);
  }

  /**
   * plot an gompertz model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_GompertzModel_2(final Path destDir,
      final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final GompertzModel model;
    final String name;

    model = new GompertzModel();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        0, 100, -0.1, 0.01d);
  }

  /**
   * plot an exp-linear model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExpLinearModelOverLogX_1(
      final Path destDir, final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final ExpLinearModelOverLogX model;
    final String name;

    model = new ExpLinearModelOverLogX();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        0, 175, -0.4d, 4);
  }

  /**
   * plot an exp-linear model for parameter step sizes
   *
   * @param destDir
   *          the destination directory
   * @param gnuplotScript
   *          the gnuplot gnuplotScript to generate
   * @param index
   *          the index
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExpLinearModelOverLogX_2(
      final Path destDir, final ITextOutput gnuplotScript, final int index)
          throws IOException {
    final ExpLinearModelOverLogX model;
    final String name;

    model = new ExpLinearModelOverLogX();
    name = _Utils._functionFileName(model, index);

    GuesserPlotter.plotModelGuesses(destDir, gnuplotScript, model, name, //
        100, -1, 0.4d, 4);
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
    final Path dest, gnuplotScriptPath;
    ITextOutput gnuplotScript;
    int index;

    dest = _Utils._destination(args, GuesserPlotter.class);
    gnuplotScriptPath = dest.resolve("gnuplot.gnuplotScript");//$NON-NLS-1$
    try (final BufferedWriter gnuplotWriter = Files
        .newBufferedWriter(gnuplotScriptPath, Charset.forName("UTF-8"))) {//$NON-NLS-1$

      gnuplotScript = AbstractTextOutput.wrap(gnuplotWriter);
      _Utils._gnuplotSetup(false, gnuplotScript, true);

      index = 0;
      GuesserPlotter.plot_ExponentialDecay_1(dest, gnuplotScript, index++);
      GuesserPlotter.plot_ExponentialDecay_2(dest, gnuplotScript, index++);
      GuesserPlotter.plot_LogisticModelWithOffsetOverLogX_1(dest,
          gnuplotScript, index++);
      GuesserPlotter.plot_LogisticModelWithOffsetOverLogX_2(dest,
          gnuplotScript, index++);
      GuesserPlotter.plot_GompertzModel_1(dest, gnuplotScript, index++);
      GuesserPlotter.plot_GompertzModel_2(dest, gnuplotScript, index++);
      GuesserPlotter.plot_ExpLinearModelOverLogX_1(dest, gnuplotScript,
          index++);
      GuesserPlotter.plot_ExpLinearModelOverLogX_2(dest, gnuplotScript,
          index++);

    }

    _Utils._gnuplotRun(dest, gnuplotScriptPath);
  }
}
