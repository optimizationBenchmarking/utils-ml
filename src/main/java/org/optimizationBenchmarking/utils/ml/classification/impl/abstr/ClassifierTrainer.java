package org.optimizationBenchmarking.utils.ml.classification.impl.abstr;

import org.optimizationBenchmarking.utils.ml.classification.spec.ClassifiedSample;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJob;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainingJobBuilder;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;
import org.optimizationBenchmarking.utils.tools.impl.abstr.Tool;

/** A classifier trainer. */
public abstract class ClassifierTrainer extends Tool
    implements IClassifierTrainer {

  /** create */
  protected ClassifierTrainer() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public IClassifierTrainingJobBuilder use() {
    this.checkCanUse();
    return new ClassifierTrainingJobBuilder(this);
  }

  /**
   * Create the training job
   *
   * @param builder
   *          the job builder
   * @return the training job
   */
  protected abstract IClassifierTrainingJob create(
      final ClassifierTrainingJobBuilder builder);

  /**
   * Create the training job
   *
   * @param builder
   *          the job builder
   * @return the training job
   */
  final IClassifierTrainingJob _create(
      final ClassifierTrainingJobBuilder builder) {
    final ClassifiedSample[] samples;
    IClassifierTrainingJob trivial;

    samples = builder.m_knownSamples;
    ClassifierTrainingJobBuilder._checkKnownSamplesNotNull(samples);
    trivial = ClassificationTools.canClusterTrivially(samples);
    if (trivial != null) {
      return trivial;
    }

    return this.create(builder);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printShortName(final ITextOutput textOut,
      final ETextCase textCase) {
    return textCase.appendWords(this.toString(), textOut);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printLongName(final ITextOutput textOut,
      final ETextCase textCase) {
    return this.printLongName(textOut, textCase);
  }

  /** {@inheritDoc} */
  @Override
  public ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    this.toText(textOut);
    return textCase.nextCase();
  }

  /** {@inheritDoc} */
  @Override
  public String getPathComponentSuggestion() {
    return this.getClass().getSimpleName();
  }
}
