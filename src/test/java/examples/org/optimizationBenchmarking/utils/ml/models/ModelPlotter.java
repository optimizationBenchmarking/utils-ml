package examples.org.optimizationBenchmarking.utils.ml.models;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.functions.UnaryFunction;
import org.optimizationBenchmarking.utils.math.functions.compound.UnaryFunctionBuilder;
import org.optimizationBenchmarking.utils.math.functions.power.Lg;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix1D;
import org.optimizationBenchmarking.utils.math.matrix.processing.FunctionSamplingJob;
import org.optimizationBenchmarking.utils.math.text.ABCParameterRenderer;
import org.optimizationBenchmarking.utils.math.text.AbstractParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExponentialDecayModel;
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
  /** the number of steps */
  private static final int STEPS = 4;

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
    function.mathRender(target, new __XRenderer());

    matrix = new FunctionSamplingJob(
        UnaryFunctionBuilder.getInstance().compound(Lg.INSTANCE, function),
        MIN_X, MAX_X, Lg.INSTANCE).call();
    size = matrix.m();
    for (index = 0; index < size; index++) {
      target.appendLineBreak();
      __render(matrix.getDouble(index, 0), target);
      target.append('\t');
      __render(matrix.getDouble(index, 1), target);
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
   * @param name
   *          the model's name
   * @param parameters
   *          the parameters
   * @param stepParameterIndex
   *          the step parameter index
   * @param stepSize
   *          the step size
   * @param destDir
   *          the destination directory
   * @throws IOException
   *           if i/o fails
   */
  public static final void plotModelStepWise(
      final ParametricUnaryFunction model, String name,
      final double[] parameters, final int stepParameterIndex,
      final double stepSize, final Path destDir) throws IOException {
    final Path path;
    final ITextOutput textOut;
    final double[] use;
    int step;

    path = PathUtils.createPathInside(destDir,
        name + '_' + stepParameterIndex + ".txt"); //$NON-NLS-1$
    use = parameters.clone();

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

        for (step = STEPS; (--step) >= 0;) {
          plotModel(model.toUnaryFunction(use), textOut);
          use[stepParameterIndex] += stepSize;
        }
      }
    }
  }

  /**
   * plot a model for parameter step sizes
   * 
   * @param destDir
   *          the destination directory
   * @throws IOException
   *           if i/o fails
   */
  public static final void plot_ExponentialDecay(final Path destDir)
      throws IOException {
    final ExponentialDecayModel model;
    double[] parameters;

    model = new ExponentialDecayModel();
    parameters = new double[] { 1, 100, -0.01,  0.7d};
    plotModelStepWise(model, model.getClass().getSimpleName() + '1',
        parameters, 0, 1d, destDir);
    plotModelStepWise(model, model.getClass().getSimpleName() + '1',
        parameters, 1, -30d, destDir);
    plotModelStepWise(model, model.getClass().getSimpleName() + '1',
        parameters, 2, -0.04d, destDir);
    plotModelStepWise(model, model.getClass().getSimpleName() + '1',
        parameters, 3, 1d, destDir);
  }

  /**
   * the main routine
   * 
   * @param args
   *          the command line arguments
   * @throws IOException
   *           if i/o fails
   */
  public static final void main(final String[] args) throws IOException {
    Path dest;

    dest = Paths.get(args[0]);
    plot_ExponentialDecay(dest);
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
    public final void renderParameter(int index, ITextOutput out) {
      if (index == 0) {
        out.append('x');
      } else {
        throwInvalidParameterIndex(index, 0);
      }
    }

    /** {@inheritDoc} */
    @Override
    public final void renderParameter(int index, IMath out) {
      if (index == 0) {
        try (final IText text = out.name()) {
          text.append('x');
        }
      } else {
        throwInvalidParameterIndex(index, 0);
      }
    }
  }

}
