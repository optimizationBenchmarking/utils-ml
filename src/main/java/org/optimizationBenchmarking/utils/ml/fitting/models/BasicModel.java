package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Arrays;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.hash.HashUtils;
import org.optimizationBenchmarking.utils.math.functions.UnaryFunction;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.DoubleConstantParameters;
import org.optimizationBenchmarking.utils.math.text.NamedMathRenderable;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.DefaultParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.ParametricUnaryFunction;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * A basic model class.
 */
public abstract class BasicModel extends ParametricUnaryFunction {

  /** create */
  protected BasicModel() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final UnaryFunction toUnaryFunction(final double[] parameters) {
    return new __InternalUnary(parameters);
  }

  /** {@inheritDoc} */
  @Override
  public IParameterGuesser createParameterGuesser(final IMatrix data) {
    return new DefaultParameterGuesser();
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final ITextOutput out,
      final IParameterRenderer renderer) {
    this.mathRender(out, renderer, new NamedMathRenderable("x")); //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer) {
    this.mathRender(out, renderer, new NamedMathRenderable("x")); //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public boolean equals(final Object o) {
    return ((o == this) || //
        ((o != null) && (o.getClass() == this.getClass())));
  }

  /** {@inheritDoc} */
  @Override
  public int hashCode() {
    return ((this.getClass().hashCode() - 1709) * 991);
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return this.getClass().getSimpleName();
  }

  /**
   * The internal unary function wrapping a specific setup of a parametric
   * unary function
   */
  private final class __InternalUnary extends UnaryFunction {
    /** the serial version uid */
    private static final long serialVersionUID = 1L;
    /** the parameters */
    private final double[] m_parameters;

    /**
     * create
     *
     * @param parameters
     *          the parameters
     */
    __InternalUnary(final double[] parameters) {
      super();
      this.m_parameters = parameters;
    }

    /** {@inheritDoc} */
    @Override
    public final int hashCode() {
      return HashUtils.combineHashes(//
          Arrays.hashCode(this.m_parameters), //
          BasicModel.this.hashCode());
    }

    /**
     * get the owner
     *
     * @return the owner
     */
    private final BasicModel __owner() {
      return BasicModel.this;
    }

    /** {@inheritDoc} */
    @Override
    public final boolean equals(final Object o) {
      final __InternalUnary x;
      if (o == this) {
        return true;
      }
      if (o instanceof __InternalUnary) {
        x = ((__InternalUnary) o);
        if (BasicModel.this.equals(x.__owner())) {
          return Arrays.equals(this.m_parameters, x.m_parameters);
        }
      }
      return false;
    }

    /** {@inheritDoc} */
    @Override
    public final double computeAsDouble(final double x0) {
      return BasicModel.this.value(x0, this.m_parameters);
    }

    /** {@inheritDoc} */
    @Override
    public final void mathRender(final ITextOutput out,
        final IParameterRenderer renderer) {
      BasicModel.this.mathRender(out, //
          new DoubleConstantParameters(this.m_parameters), //
          new __SingleRenderable(renderer));
    }

    /** {@inheritDoc} */
    @Override
    public final void mathRender(final IMath out,
        final IParameterRenderer renderer) {
      BasicModel.this.mathRender(out, //
          new DoubleConstantParameters(this.m_parameters), //
          new __SingleRenderable(renderer));
    }
  }

  /** the single renderable */
  private static final class __SingleRenderable
      implements IMathRenderable {

    /** the renderer */
    private final IParameterRenderer m_render;

    /**
     * create
     *
     * @param render
     *          the parameter renderer
     */
    __SingleRenderable(final IParameterRenderer render) {
      super();
      this.m_render = render;
    }

    /** {@inheritDoc} */
    @Override
    public final void mathRender(final ITextOutput out,
        final IParameterRenderer renderer) {
      this.m_render.renderParameter(0, out);
    }

    /** {@inheritDoc} */
    @Override
    public final void mathRender(final IMath out,
        final IParameterRenderer renderer) {
      this.m_render.renderParameter(0, out);
    }
  }
}
