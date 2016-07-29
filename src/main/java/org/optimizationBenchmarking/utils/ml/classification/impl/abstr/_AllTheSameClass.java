package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** All the samples belong to the same class */
final class _AllTheSameClass extends _ImmediateClassifier {

  /** the class */
  private final int m_clazz;

  /**
   * create the all-the-same-class classifier
   *
   * @param clazz
   *          the classifier
   */
  _AllTheSameClass(final int clazz) {
    super();
    this.m_clazz = clazz;
  }

  /** {@inheritDoc} */
  @Override
  public final int classify(final double[] features) {
    return this.m_clazz;
  }

  /** {@inheritDoc} */
  @Override
  public final void render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    if (textOutput instanceof IComplexText) {
      try (final IText text = ((IComplexText) textOutput).inlineCode()) {
        this.__render(renderer, text);
      }
    } else {
      this.__render(renderer, textOutput);
    }
  }

  /**
   * Render the classifier to a given text output destination.
   *
   * @param renderer
   *          the renderer
   * @param textOutput
   *          the text output destination
   */
  private final void __render(final IClassifierParameterRenderer renderer,
      final ITextOutput textOutput) {
    textOutput.append("Always class "); //$NON-NLS-1$
    renderer.renderShortClassName(this.m_clazz, textOutput);
  }
}
