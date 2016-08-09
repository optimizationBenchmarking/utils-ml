package org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased;

import java.net.URI;

import org.optimizationBenchmarking.utils.bibliography.data.BibArticle;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthor;
import org.optimizationBenchmarking.utils.bibliography.data.BibAuthors;
import org.optimizationBenchmarking.utils.bibliography.data.BibBook;
import org.optimizationBenchmarking.utils.bibliography.data.BibDate;
import org.optimizationBenchmarking.utils.bibliography.data.BibInProceedings;
import org.optimizationBenchmarking.utils.bibliography.data.BibOrganization;
import org.optimizationBenchmarking.utils.bibliography.data.BibProceedings;
import org.optimizationBenchmarking.utils.bibliography.data.BibTechReport;
import org.optimizationBenchmarking.utils.bibliography.data.BibliographyBuilder;
import org.optimizationBenchmarking.utils.bibliography.data.EBibMonth;
import org.optimizationBenchmarking.utils.document.spec.ECitationMode;
import org.optimizationBenchmarking.utils.document.spec.IComplexText;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.math.mathEngine.impl.R.R;
import org.optimizationBenchmarking.utils.ml.clustering.impl.abstr.ClusteringJob;
import org.optimizationBenchmarking.utils.ml.clustering.impl.abstr.DistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.impl.abstr.DistanceClusteringJobBuilder;
import org.optimizationBenchmarking.utils.text.ESequenceMode;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** The {@code R}-based distance distance clustering engine. */
public final class RBasedDistanceClusterer extends DistanceClusterer {

  /** the method name */
  static final String METHOD = "R-based Distance Clusterer"; //$NON-NLS-1$

  /** the organization */
  private static final BibOrganization ORGANIZATION_R = new BibOrganization(//
      "The R Project for Statistical Computing, The R Foundation", //$NON-NLS-1$
      "Vienna, Austria", null); //$NON-NLS-1$

  /** the R reference 1 */
  private static final BibTechReport REFERENCE_R_1 = new BibTechReport( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Martin", "M\u00e4chler"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Peter", "Rousseeuw"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Anja", "Struyf"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Mia", "Hubert"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Kurt", "Hornik"), //$NON-NLS-1$//$NON-NLS-2$
  }), "cluster: Cluster Analysis Basics and Extensions", //$NON-NLS-1$
      new BibDate(2015), //
      null, //
      null, //
      null, //
      RBasedDistanceClusterer.ORGANIZATION_R, //
      URI.create(
          "https://cran.r-project.org/web/packages/cluster/cluster.pdf"), //$NON-NLS-1$
      null);

  /** the R reference 2 */
  private static final BibTechReport REFERENCE_R_2 = new BibTechReport( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Christian", "Henning"), //$NON-NLS-1$//$NON-NLS-2$
  }), "fpc: Flexible Procedures for Clustering", //$NON-NLS-1$
      new BibDate(2015), //
      null, //
      null, //
      null, //
      RBasedDistanceClusterer.ORGANIZATION_R, //
      URI.create("https://cran.r-project.org/web/packages/fpc/fpc.pdf"), //$NON-NLS-1$
      null);

  /** The first PAM reference */
  private static final BibArticle REFERENCE_PAM_1 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Alan P.", "Reynolds"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Graeme", "Richards"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Beatriz", "de la Iglesia"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Victor J.", "Rayward-Smith"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Clustering Rules: A Comparison of Partitioning and Hierarchical Clustering Algorithms", //$NON-NLS-1$
      new BibDate(2006, EBibMonth.DECEMBER),
      "Journal of Mathematical Modelling and Algorithms", //$NON-NLS-1$
      "1570-1166", //$NON-NLS-1$
      "5", //$NON-NLS-1$
      "4", //$NON-NLS-1$
      "475", //$NON-NLS-1$
      "504", //$NON-NLS-1$
      new BibOrganization(//
          "Springer International Publishing AG", //$NON-NLS-1$
          "Cham, Switzerland", null), //$NON-NLS-1$
      null, "10.1007/s10852-005-9022-1");//$NON-NLS-1$

  /** The second PAM reference */
  private static final BibArticle REFERENCE_PAM_2 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("T.", "Cali\u0144ski"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("J.", "Harabasz"), //$NON-NLS-1$//$NON-NLS-2$
  }), "A Dendrite Method for Cluster Analysis", //$NON-NLS-1$
      new BibDate(1974),
      "Communications in Statistics \u2012 Theory and Methods", //$NON-NLS-1$
      "0361-0926", //$NON-NLS-1$
      "3", //$NON-NLS-1$
      "1", //$NON-NLS-1$
      "1", //$NON-NLS-1$
      "27", //$NON-NLS-1$
      new BibOrganization(//
          "Taylor & Francis", //$NON-NLS-1$
          "Milton Park, Abingdon-on-Thames, Oxfordshire, UK", null), //$NON-NLS-1$
      null, "10.1080/03610927408827101");//$NON-NLS-1$

  /** The third PAM reference */
  private static final BibArticle REFERENCE_PAM_3 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Christian", "Henning"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Tim F.", "Liao"), //$NON-NLS-1$//$NON-NLS-2$
  }), "How to Find an Appropriate Clustering for Mixed-Type Variables with Application to Socio-Economic Stratification", //$NON-NLS-1$
      new BibDate(2003, EBibMonth.MAY),
      "Journal of the Royal Statistical Society: Series C (Applied Statistics)", //$NON-NLS-1$
      "1467-9876", //$NON-NLS-1$
      "62", //$NON-NLS-1$
      "3", //$NON-NLS-1$
      "309", //$NON-NLS-1$
      "369", //$NON-NLS-1$
      new BibOrganization(//
          "John Wiley & Sons, Inc.", //$NON-NLS-1$
          "Hoboken, NJ, USA", null), //$NON-NLS-1$
      null, "10.1111/j.1467-9876.2012.01066.x");//$NON-NLS-1$

  /** The first hierarchical clustering reference */
  private static final BibArticle REFERENCE_HIERARCHICAL_1 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Fionn", "Murtagh"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Pierre", "Legendre"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward's Criterion? ", //$NON-NLS-1$
      new BibDate(2014, EBibMonth.OCTOBER), "Journal of Classification", //$NON-NLS-1$
      "0176-4268", //$NON-NLS-1$
      "31", //$NON-NLS-1$
      "3", //$NON-NLS-1$
      "274", //$NON-NLS-1$
      "295", //$NON-NLS-1$
      new BibOrganization(//
          "Springer International Publishing AG", //$NON-NLS-1$
          "Cham, Switzerland", null), //$NON-NLS-1$
      null, "10.1007%2Fs00357-014-9161-z");//$NON-NLS-1$

  /** the second hierarchical clustering reference */
  private static final BibBook REFERENCE_HIERARCHICAL_2 = new BibBook( //
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("John A.", "Hartigan"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Clustering Algorithms (Probability & Mathematical Statistics)", //$NON-NLS-1$
      new BibDate(1975, EBibMonth.APRIL), //
      BibAuthors.EMPTY_AUTHORS, //
      new BibOrganization(//
          "John Wiley & Sons, Inc.", //$NON-NLS-1$
          "Hoboken, NJ, USA", null), //$NON-NLS-1$
      null, //
      null, //
      null, //
      null, //
      "047135645X", //$NON-NLS-1$
      null, //
      null);

  /** The DBSCAN reference */
  private static final BibInProceedings REFERENCE_DBSCAN = new BibInProceedings(//
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Martin", "Ester"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Hans-Peter", "Kriegel"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("J\u00f6rg", "Sander"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Xiaowei", "Xu"), //$NON-NLS-1$//$NON-NLS-2$
  }), "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", //$NON-NLS-1$
      new BibProceedings( //
          "Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)", //$NON-NLS-1$
          new BibDate(1996), //
          new BibDate(1996), //
          new BibAuthors(new BibAuthor[] { //
              new BibAuthor("Evangelos", "Simoudis"), //$NON-NLS-1$//$NON-NLS-2$
              new BibAuthor("Jiawei", "Han"), //$NON-NLS-1$//$NON-NLS-2$
              new BibAuthor("Usama M.", "Fayyad"), //$NON-NLS-1$//$NON-NLS-2$
  }), new BibOrganization(null, "Portland, Oregon, USA", null), //$NON-NLS-1$
          new BibOrganization(//
              "AAAI Press", //$NON-NLS-1$
              "Palo Alto, CA, USA", null), //$NON-NLS-1$
          null, //
          null, //
          null, //
          "1-57735-004-9", //$NON-NLS-1$
          URI.create("http://www.aaai.org/Library/KDD/kdd96contents.php"), //$NON-NLS-1$
          null), //
      "226", //$NON-NLS-1$
      "231", //$NON-NLS-1$
      null, //
      URI.create("http://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf"), //$NON-NLS-1$
      null);

  /** The first silhouette reference */
  private static final BibArticle REFERENCE_SILHOUETTE_1 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Peter J.", "Rousseeuw"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis", //$NON-NLS-1$
      new BibDate(1987, EBibMonth.NOVEMBER),
      "Computational and Applied Mathematics", //$NON-NLS-1$
      "0377-0427", //$NON-NLS-1$
      "20", //$NON-NLS-1$
      null, //
      "53", //$NON-NLS-1$
      "65", //$NON-NLS-1$
      new BibOrganization(//
          "Elsevier Science Publishers B.V.", //$NON-NLS-1$
          "Amsterdam, The Netherlands", null), //$NON-NLS-1$
      null, "10.1016/0377-0427(87)90125-7");//$NON-NLS-1$

  /** The second silhouette reference */
  private static final BibArticle REFERENCE_SILHOUETTE_2 = new BibArticle(
      new BibAuthors(new BibAuthor[] { //
          new BibAuthor("Renato Cordeiro", "de Amorim"), //$NON-NLS-1$//$NON-NLS-2$
          new BibAuthor("Christian", "Henning"), //$NON-NLS-1$//$NON-NLS-2$
  }), "Recovering the Number of Clusters in Data Sets with Noise Features using Feature Rescaling Factors", //$NON-NLS-1$
      new BibDate(2015, EBibMonth.DECEMBER, 10), "Information Sciences", //$NON-NLS-1$
      "0020-0255", //$NON-NLS-1$
      "324", //$NON-NLS-1$
      null, //
      "126", //$NON-NLS-1$
      "145", //$NON-NLS-1$
      new BibOrganization(//
          "Elsevier Science Publishers B.V.", //$NON-NLS-1$
          "Amsterdam, The Netherlands", null), //$NON-NLS-1$
      null, "10.1016/j.ins.2015.06.039");//$NON-NLS-1$

  /** create */
  RBasedDistanceClusterer() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final boolean canUse() {
    return R.getInstance().canUse();
  }

  /** {@inheritDoc} */
  @Override
  public final void checkCanUse() {
    R.getInstance().checkCanUse();
    super.checkCanUse();
  }

  /** {@inheritDoc} */
  @Override
  protected final ClusteringJob create(
      final DistanceClusteringJobBuilder builder) {
    return new _RBasedDistanceClusteringJob(builder);
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return RBasedDistanceClusterer.METHOD;
  }

  /** {@inheritDoc} */
  @Override
  public final ETextCase printDescription(final ITextOutput textOut,
      final ETextCase textCase) {
    IComplexText complex;
    ETextCase next;

    next = textCase.appendWord("several", textOut); //$NON-NLS-1$

    textOut.append(" clustering algorithms implemented in "); //$NON-NLS-1$
    if (textOut instanceof IComplexText) {
      complex = ((IComplexText) textOut);
      try (final IText code = complex.inlineCode()) {
        code.append('R');
      }
      try (final BibliographyBuilder builder = complex
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(RBasedDistanceClusterer.REFERENCE_R_1);
        builder.add(RBasedDistanceClusterer.REFERENCE_R_2);
      }
    } else {
      complex = null;
      textOut.append('R');
    }

    textOut.append(", including Partitioning Around Medoids (PAM, PAMk)"); //$NON-NLS-1$
    if (complex != null) {
      try (final BibliographyBuilder builder = complex
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(RBasedDistanceClusterer.REFERENCE_PAM_1);
        builder.add(RBasedDistanceClusterer.REFERENCE_PAM_2);
        builder.add(RBasedDistanceClusterer.REFERENCE_PAM_3);
      }

    }
    textOut.append(", hierarchical clustering"); //$NON-NLS-1$
    if (complex != null) {
      try (final BibliographyBuilder builder = complex
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(RBasedDistanceClusterer.REFERENCE_HIERARCHICAL_1);
        builder.add(RBasedDistanceClusterer.REFERENCE_HIERARCHICAL_2);
      }
    }

    textOut.append(" with different agglomeration methods, and DBSCAN"); //$NON-NLS-1$
    if (complex != null) {
      try (final BibliographyBuilder builder = complex
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(RBasedDistanceClusterer.REFERENCE_DBSCAN);
      }
    }

    textOut.append(
        ", are applied and the clustering result with the best average silhouette width"); //$NON-NLS-1$
    if (complex != null) {
      try (final BibliographyBuilder builder = complex
          .cite(ECitationMode.ID, next, ESequenceMode.COMMA)) {
        builder.add(RBasedDistanceClusterer.REFERENCE_SILHOUETTE_1);
        builder.add(RBasedDistanceClusterer.REFERENCE_SILHOUETTE_2);
      }
    }

    textOut.append(" is returned."); //$NON-NLS-1$

    return next.nextCase();
  }

  /**
   * Get the globally shared instance of the {@code R}-based distance
   * clusterer.
   *
   * @return the globally shared instance of the {@code R}-based distance
   *         clusterer.
   */
  public static final RBasedDistanceClusterer getInstance() {
    return __RBasedClustererHolder.INSTANCE;
  }

  /** the clusterer holder */
  private static final class __RBasedClustererHolder {
    /** the globally shared instance */
    static final RBasedDistanceClusterer INSTANCE = new RBasedDistanceClusterer();
  }
}
