package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.optimizationBenchmarking.utils.math.matrix.AbstractMatrix;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.MatrixBuilder;
import org.optimizationBenchmarking.utils.ml.fitting.models.ExponentialDecayModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.GompertzModel;
import org.optimizationBenchmarking.utils.ml.fitting.models.LogisticModelWithOffsetOverLogX;
import org.optimizationBenchmarking.utils.ml.fitting.models.QuadraticModel;
import org.optimizationBenchmarking.utils.text.TextUtils;

import shared.junit.org.optimizationBenchmarking.utils.ml.fitting.FittingExampleDataset;

/** The fitting example datasets. */
public final class FittingExampleDatasets {

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset A_1FlipHC_uf020_01_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset B_mFlipHC_uf100_01_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset C_2FlipHCrs_uf250_01_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_01_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_02_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_03_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_04_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_05_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_06_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_07_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_08_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_09_FTQ;

  /** the example: fes/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_FOL;
  /** the example: fes/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_FOE;
  /** the example: fes/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_FOG;
  /** the example: t/o logistics model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_TOL;
  /** the example: t/o decay model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_TOE;
  /** the example: t/o gompertz model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_TOG;
  /** the example: t/fes quadratic model */
  public static final FittingExampleDataset D_2FlipHC_uf250_10_FTQ;

  static {
    FittingExampleDatasets sets;

    sets = new FittingExampleDatasets();
    sets.__loadMatrix("1FlipHC-uf020-01"); //$NON-NLS-1$
    A_1FlipHC_uf020_01_FOL = sets.__fol();
    A_1FlipHC_uf020_01_FOE = sets.__foe();
    A_1FlipHC_uf020_01_FOG = sets.__fog();
    A_1FlipHC_uf020_01_TOL = sets.__tol();
    A_1FlipHC_uf020_01_TOE = sets.__toe();
    A_1FlipHC_uf020_01_TOG = sets.__tog();
    A_1FlipHC_uf020_01_FTQ = sets.__ftq();

    sets.__loadMatrix("mFlipHC-uf100-01"); //$NON-NLS-1$
    B_mFlipHC_uf100_01_FOL = sets.__fol();
    B_mFlipHC_uf100_01_FOE = sets.__foe();
    B_mFlipHC_uf100_01_FOG = sets.__fog();
    B_mFlipHC_uf100_01_TOL = sets.__tol();
    B_mFlipHC_uf100_01_TOE = sets.__toe();
    B_mFlipHC_uf100_01_TOG = sets.__tog();
    B_mFlipHC_uf100_01_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHCrs-uf250-01"); //$NON-NLS-1$
    C_2FlipHCrs_uf250_01_FOL = sets.__fol();
    C_2FlipHCrs_uf250_01_FOE = sets.__foe();
    C_2FlipHCrs_uf250_01_FOG = sets.__fog();
    C_2FlipHCrs_uf250_01_TOL = sets.__tol();
    C_2FlipHCrs_uf250_01_TOE = sets.__toe();
    C_2FlipHCrs_uf250_01_TOG = sets.__tog();
    C_2FlipHCrs_uf250_01_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-01"); //$NON-NLS-1$
    D_2FlipHC_uf250_01_FOL = sets.__fol();
    D_2FlipHC_uf250_01_FOE = sets.__foe();
    D_2FlipHC_uf250_01_FOG = sets.__fog();
    D_2FlipHC_uf250_01_TOL = sets.__tol();
    D_2FlipHC_uf250_01_TOE = sets.__toe();
    D_2FlipHC_uf250_01_TOG = sets.__tog();
    D_2FlipHC_uf250_01_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-02"); //$NON-NLS-1$
    D_2FlipHC_uf250_02_FOL = sets.__fol();
    D_2FlipHC_uf250_02_FOE = sets.__foe();
    D_2FlipHC_uf250_02_FOG = sets.__fog();
    D_2FlipHC_uf250_02_TOL = sets.__tol();
    D_2FlipHC_uf250_02_TOE = sets.__toe();
    D_2FlipHC_uf250_02_TOG = sets.__tog();
    D_2FlipHC_uf250_02_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-03"); //$NON-NLS-1$
    D_2FlipHC_uf250_03_FOL = sets.__fol();
    D_2FlipHC_uf250_03_FOE = sets.__foe();
    D_2FlipHC_uf250_03_FOG = sets.__fog();
    D_2FlipHC_uf250_03_TOL = sets.__tol();
    D_2FlipHC_uf250_03_TOE = sets.__toe();
    D_2FlipHC_uf250_03_TOG = sets.__tog();
    D_2FlipHC_uf250_03_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-04"); //$NON-NLS-1$
    D_2FlipHC_uf250_04_FOL = sets.__fol();
    D_2FlipHC_uf250_04_FOE = sets.__foe();
    D_2FlipHC_uf250_04_FOG = sets.__fog();
    D_2FlipHC_uf250_04_TOL = sets.__tol();
    D_2FlipHC_uf250_04_TOE = sets.__toe();
    D_2FlipHC_uf250_04_TOG = sets.__tog();
    D_2FlipHC_uf250_04_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-05"); //$NON-NLS-1$
    D_2FlipHC_uf250_05_FOL = sets.__fol();
    D_2FlipHC_uf250_05_FOE = sets.__foe();
    D_2FlipHC_uf250_05_FOG = sets.__fog();
    D_2FlipHC_uf250_05_TOL = sets.__tol();
    D_2FlipHC_uf250_05_TOE = sets.__toe();
    D_2FlipHC_uf250_05_TOG = sets.__tog();
    D_2FlipHC_uf250_05_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-06"); //$NON-NLS-1$
    D_2FlipHC_uf250_06_FOL = sets.__fol();
    D_2FlipHC_uf250_06_FOE = sets.__foe();
    D_2FlipHC_uf250_06_FOG = sets.__fog();
    D_2FlipHC_uf250_06_TOL = sets.__tol();
    D_2FlipHC_uf250_06_TOE = sets.__toe();
    D_2FlipHC_uf250_06_TOG = sets.__tog();
    D_2FlipHC_uf250_06_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-07"); //$NON-NLS-1$
    D_2FlipHC_uf250_07_FOL = sets.__fol();
    D_2FlipHC_uf250_07_FOE = sets.__foe();
    D_2FlipHC_uf250_07_FOG = sets.__fog();
    D_2FlipHC_uf250_07_TOL = sets.__tol();
    D_2FlipHC_uf250_07_TOE = sets.__toe();
    D_2FlipHC_uf250_07_TOG = sets.__tog();
    D_2FlipHC_uf250_07_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-08"); //$NON-NLS-1$
    D_2FlipHC_uf250_08_FOL = sets.__fol();
    D_2FlipHC_uf250_08_FOE = sets.__foe();
    D_2FlipHC_uf250_08_FOG = sets.__fog();
    D_2FlipHC_uf250_08_TOL = sets.__tol();
    D_2FlipHC_uf250_08_TOE = sets.__toe();
    D_2FlipHC_uf250_08_TOG = sets.__tog();
    D_2FlipHC_uf250_08_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-09"); //$NON-NLS-1$
    D_2FlipHC_uf250_09_FOL = sets.__fol();
    D_2FlipHC_uf250_09_FOE = sets.__foe();
    D_2FlipHC_uf250_09_FOG = sets.__fog();
    D_2FlipHC_uf250_09_TOL = sets.__tol();
    D_2FlipHC_uf250_09_TOE = sets.__toe();
    D_2FlipHC_uf250_09_TOG = sets.__tog();
    D_2FlipHC_uf250_09_FTQ = sets.__ftq();

    sets.__loadMatrix("2FlipHC-uf250-10"); //$NON-NLS-1$
    D_2FlipHC_uf250_10_FOL = sets.__fol();
    D_2FlipHC_uf250_10_FOE = sets.__foe();
    D_2FlipHC_uf250_10_FOG = sets.__fog();
    D_2FlipHC_uf250_10_TOL = sets.__tol();
    D_2FlipHC_uf250_10_TOE = sets.__toe();
    D_2FlipHC_uf250_10_TOG = sets.__tog();
    D_2FlipHC_uf250_10_FTQ = sets.__ftq();
  }

  /** the time-objective model 1 */
  private final LogisticModelWithOffsetOverLogX m_modelLogistic;
  /** the time-objective model 2 */
  private final ExponentialDecayModel m_modelDecay;
  /** the time-objective model 3 */
  private final GompertzModel m_modelGompertz;
  /** the time-time model 2 */
  private final QuadraticModel m_modelQuadratic;

  /** the name */
  private String m_name;

  /** the current matrix. */
  private AbstractMatrix m_matrix;

  /** the f-o matrix */
  private IMatrix m_fo;
  /** the t-o matrix */
  private IMatrix m_to;
  /** the f-t matrix */
  private IMatrix m_ft;

  /** the forbidden constructor */
  private FittingExampleDatasets() {
    super();
    this.m_modelLogistic = new LogisticModelWithOffsetOverLogX();
    this.m_modelDecay = new ExponentialDecayModel();
    this.m_modelGompertz = new GompertzModel();
    this.m_modelQuadratic = new QuadraticModel();
  }

  /**
   * the f-o matrix
   *
   * @return the matrix
   */
  private final IMatrix __fo() {
    if (this.m_fo == null) {
      this.m_fo = this.m_matrix.selectColumns(0, 2).copy();
    }
    return this.m_fo;
  }

  /**
   * the t-o matrix
   *
   * @return the matrix
   */
  private final IMatrix __to() {
    if (this.m_to == null) {
      this.m_to = this.m_matrix.selectColumns(1, 2).copy();
    }
    return this.m_to;
  }

  /**
   * the f-t matrix
   *
   * @return the matrix
   */
  private final IMatrix __ft() {
    if (this.m_ft == null) {
      this.m_ft = this.m_matrix.selectColumns(0, 1).copy();
    }
    return this.m_ft;
  }

  /**
   * create the logistic model example
   *
   * @return the example
   */
  private final FittingExampleDataset __fol() {
    return new FittingExampleDataset(this.m_name + "_fol", //$NON-NLS-1$
        this.__fo(), this.m_modelLogistic);
  }

  /**
   * create the exponential decay example
   *
   * @return the example
   */
  private final FittingExampleDataset __foe() {
    return new FittingExampleDataset(this.m_name + "_foe", //$NON-NLS-1$
        this.__fo(), this.m_modelDecay);
  }

  /**
   * create the logistic model example
   *
   * @return the example
   */
  private final FittingExampleDataset __tol() {
    return new FittingExampleDataset(this.m_name + "_tol", //$NON-NLS-1$
        this.__to(), this.m_modelLogistic);
  }

  /**
   * create the exponential decay example
   *
   * @return the example
   */
  private final FittingExampleDataset __toe() {
    return new FittingExampleDataset(this.m_name + "_toe", //$NON-NLS-1$
        this.__to(), this.m_modelDecay);
  }

  /**
   * create the Gompertz model example
   *
   * @return the example
   */
  private final FittingExampleDataset __fog() {
    return new FittingExampleDataset(this.m_name + "_fog", //$NON-NLS-1$
        this.__fo(), this.m_modelGompertz);
  }

  /**
   * create the Gompertz decay example
   *
   * @return the example
   */
  private final FittingExampleDataset __tog() {
    return new FittingExampleDataset(this.m_name + "_tog", //$NON-NLS-1$
        this.__to(), this.m_modelGompertz);
  }

  /**
   * create the quadractic
   *
   * @return the example
   */
  private final FittingExampleDataset __ftq() {
    return new FittingExampleDataset(this.m_name + "_ftq", //$NON-NLS-1$
        this.__ft(), this.m_modelQuadratic);
  }

  /**
   * load a data matrix example
   *
   * @param resource
   *          the resource
   */
  private final void __loadMatrix(final String resource) {
    final MatrixBuilder builder;
    int i, j;
    String str;

    this.m_matrix = null;
    this.m_fo = null;
    this.m_ft = null;
    this.m_to = null;
    this.m_name = resource;

    builder = new MatrixBuilder();
    builder.setN(3);

    try (InputStream is = this.getClass()
        .getResourceAsStream(resource + ".txt")) { //$NON-NLS-1$
      try (InputStreamReader isr = new InputStreamReader(is)) {
        try (BufferedReader br = new BufferedReader(isr)) {
          while ((str = br.readLine()) != null) {
            if ((str = TextUtils.prepare(str)) == null) {
              continue;
            }

            i = str.indexOf('\t');

            builder.append(Double.parseDouble(str.substring(0, i)));

            i++;
            j = str.indexOf('\t', i);
            builder.append(Double.parseDouble(str.substring(i, j)));

            builder.append(Double.parseDouble(str.substring(j + 1)));
          }
        }
      }
    } catch (final Throwable error) {
      throw new RuntimeException(error);
    }

    this.m_matrix = builder.make();
  }
}
