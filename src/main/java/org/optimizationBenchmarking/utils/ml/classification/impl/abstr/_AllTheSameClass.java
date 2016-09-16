package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifier;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierParameterRenderer;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingResult;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** All the samples belong to the same class */
final class _AllTheSameClass extends Classifier
    implements IClassifierTrainingJob, IClassifierTrainingResult {

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
  public final double getComplexity() {
    return 0d;
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

  /** {@inheritDoc} */
  @Override
  public ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWords("fixed class", textOut); //$NON-NLS-1$
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
    textOutput.append("The classifier always returns class "); //$NON-NLS-1$
    renderer.renderShortClassName(this.m_clazz, textOutput);
    textOutput.append('.');
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifierTrainingResult call() {
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public final IClassifier getClassifier() {
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public final double getQuality() {
    return 0;
  }
}
