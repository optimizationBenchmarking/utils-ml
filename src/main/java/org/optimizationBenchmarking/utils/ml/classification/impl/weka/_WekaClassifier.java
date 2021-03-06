package org.optimizationBenchmarking.utils.ml.classification.impl.weka;

import java.net.URI;
import java.util.Arrays;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.ml.classification.impl.abstr.SimplifyingClassifier;

import weka.classifiers.Classifier;

/**
 * The base class for wrapping Weka classifiers
 *
 * @param <CT>
 *          the classifier type
 */
abstract class _WekaClassifier<CT extends Classifier>
    extends SimplifyingClassifier {

  /** The article for citing weka */
  static final BibArticle WEKA = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Mark", "Hall"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Eibe", "Frank"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Geoffrey", "Holmes"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Bernhard", "Pfahringer"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Peter", "Reutemann"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Ian H.", "Witten"),//$NON-NLS-1$//$NON-NLS-2$
  }), "The WEKA Data Mining Software: An Update", //$NON-NLS-1$
      new BibDate(2009, EBibMonth.JUNE), "SIGKDD Explorations", //$NON-NLS-1$
      "1931-0145", //$NON-NLS-1$
      "11", //$NON-NLS-1$
      "1", //$NON-NLS-1$
      "10", //$NON-NLS-1$
      "18", //$NON-NLS-1$
      new BibOrganization(//
          "ACM", //$NON-NLS-1$
          "New York, NY, USA", null), //$NON-NLS-1$
      URI.create(
          "http://www.cms.waikato.ac.nz/~ml/publications/2009/weka_update.pdf"), //$NON-NLS-1$
      "10.1145/1656274.1656278");//$NON-NLS-1$

  /** the internal classifier */
  final CT m_classifier;

  /** the instance to use */
  private final _InternalInstance m_instance;

  /**
   * Create the weka classifier wrapper
   *
   * @param selectedFeatures
   *          the selected features
   * @param classifier
   *          the classifier
   * @param instance
   *          to use
   */
  _WekaClassifier(final int[] selectedFeatures, final CT classifier,
      final _InternalInstance instance) {
    super(selectedFeatures);
    if (classifier == null) {
      throw new IllegalArgumentException("Classifier must not be null."); //$NON-NLS-1$
    }
    if (instance == null) {
      throw new IllegalArgumentException("Instance must not be null."); //$NON-NLS-1$
    }

    this.m_classifier = classifier;
    this.m_instance = instance;
  }

  /** {@inheritDoc} */
  @Override
  public final int classify(final double[] features) {
    this.m_instance._assign(features, this.m_selectedFeatures);

    try {
      return ((int) (0.5d
          + this.m_classifier.classifyInstance(this.m_instance)));
    } catch (final Exception exception) {
      throw new IllegalArgumentException(
          "Error when trying to classify instance " //$NON-NLS-1$
              + Arrays.toString(features),
          exception);
    }
  }
}
