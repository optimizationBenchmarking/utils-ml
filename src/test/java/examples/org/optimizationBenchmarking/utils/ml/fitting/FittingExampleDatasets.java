package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.matrix.impl.DoubleMatrix2D;
import org.optimizationBenchmarking.utils.parallel.Execute;
import org.optimizationBenchmarking.utils.text.TextUtils;

import shared.junit.TestBase;

/** Some examples for data fitting. */
public final class FittingExampleDatasets extends TestBase
    implements Callable<ArrayListView<FittingExampleDataset>> {

  /** the resources */
  private static final String[] RESOURCES = { //
      "1FlipHC-uf020-01.txt", //$NON-NLS-1$
      "mFlipHC-uf100-01.txt", //$NON-NLS-1$
      "2FlipHCrs-uf250-01.txt", //$NON-NLS-1$
      "2FlipHC-uf250-01.txt", //$NON-NLS-1$
      "2FlipHC-uf250-02.txt", //$NON-NLS-1$
      "2FlipHC-uf250-03.txt", //$NON-NLS-1$
      "2FlipHC-uf250-04.txt", //$NON-NLS-1$
      "2FlipHC-uf250-05.txt", //$NON-NLS-1$
      "2FlipHC-uf250-06.txt", //$NON-NLS-1$
      "2FlipHC-uf250-07.txt", //$NON-NLS-1$
      "2FlipHC-uf250-08.txt", //$NON-NLS-1$
      "2FlipHC-uf250-09.txt", //$NON-NLS-1$
      "2FlipHC-uf250-10.txt", //$NON-NLS-1$
  };

  /** the internal sorter */
  private final __Sorter m_sorter;

  /** the logger */
  final Logger m_logger;

  /** allow time-time and objective-objective data sets? */
  private final boolean m_useSameType;

  /**
   * create
   *
   * @param logger
   *          the logger
   * @param useSameType
   *          allow time-time and objective-objective data sets?
   */
  public FittingExampleDatasets(final Logger logger,
      final boolean useSameType) {
    super();

    this.m_sorter = new __Sorter();
    this.m_logger = logger;
    this.m_useSameType = useSameType;
  }

  /** {@inheritDoc} */
  @Override
  public final ArrayListView<FittingExampleDataset> call()
      throws IOException {
    final ArrayListView<FittingExampleDataset> result;
    final ArrayList<Future<Void>> jobs;
    ArrayList<FittingExampleDataset> list;
    FittingExampleDataset[] sets;

    if ((this.m_logger != null)
        && (this.m_logger.isLoggable(Level.FINE))) {
      this.m_logger.fine("Begin loading example datasets."); //$NON-NLS-1$
    }

    list = new ArrayList<>();
    jobs = new ArrayList<>();

    for (final String resource : FittingExampleDatasets.RESOURCES) {
      jobs.add(Execute.parallel(new __AppendResource(resource, list)));
    }

    Execute.join(jobs);

    sets = list.toArray(new FittingExampleDataset[list.size()]);
    Arrays.sort(sets);
    result = new ArrayListView<>(sets);

    list.clear();
    list = null;

    if ((this.m_logger != null)
        && (this.m_logger.isLoggable(Level.FINE))) {
      this.m_logger.fine("Finished loading example datasets."); //$NON-NLS-1$
    }

    return result;
  }

  /**
   * append a given resource
   *
   * @param resource
   *          the resource
   * @param list
   *          the list
   * @throws IOException
   *           if i/o fails
   */
  final void _appendResource(final String resource,
      final ArrayList<FittingExampleDataset> list) throws IOException {
    final ArrayList<IMatrix> loaded;
    final ArrayList<double[]> current;
    int i, j, size;
    String str;
    double[] data;

    loaded = new ArrayList<>();
    current = new ArrayList<>();
    try (InputStream is = this.getClass().getResourceAsStream(resource)) {
      try (InputStreamReader isr = new InputStreamReader(is)) {
        try (BufferedReader br = new BufferedReader(isr)) {
          while ((str = br.readLine()) != null) {
            if ((str = TextUtils.prepare(str)) == null) {
              size = current.size();
              if (size > 0) {
                loaded.add(new DoubleMatrix2D(
                    current.toArray(new double[size][])));
              }
              current.clear();
              continue;
            }
            data = new double[3];

            i = str.indexOf('\t');
            data[0] = Double.parseDouble(str.substring(0, i));

            i++;
            j = str.indexOf('\t', i);
            data[1] = Double.parseDouble(str.substring(i, j));

            data[2] = Double.parseDouble(str.substring(j + 1));
            current.add(data);
          }
        }
      }
    }

    size = current.size();
    if (size > 0) {
      loaded.add(new DoubleMatrix2D(current.toArray(new double[size][])));
    }

    if (loaded.size() > 0) {
      this.__postprocess(
          ("maxSat_" + //$NON-NLS-1$
              PathUtils.getFileNameWithoutExtension(resource)), //
          loaded, list, new int[] { -1, -1, 1 });
    }
  }

  /**
   * Post-process the loaded data
   *
   * @param name
   *          the name of the example data set
   * @param loaded
   *          the data
   * @param list
   *          the list to append to
   * @param dimTypes
   *          the description of the dimensions: {@code -1} for time
   *          dimensions, {@code 1} for objective dimensions, {@code 0} for
   *          skip
   */
  private final void __postprocess(final String name,
      final Collection<? extends IMatrix> loaded,
      final ArrayList<FittingExampleDataset> list, final int[] dimTypes) {
    FittingExampleDataset ds;
    int dimA, dimB, useDimA, useDimB, total, index, size, row;
    double[][] rawPoints;
    String useName;

    total = 0;
    for (final IMatrix current : loaded) {
      total += current.m();
    }

    for (dimA = 0; dimA < dimTypes.length; dimA++) {
      if (dimTypes[dimA] == 0) {
        continue;
      }
      for (dimB = (dimA + 1); dimB < dimTypes.length; dimB++) {
        if (dimTypes[dimB] == 0) {
          continue;
        }

        if ((!this.m_useSameType) && (dimTypes[dimA] == dimTypes[dimB])) {
          continue;
        }

        if ((dimTypes[dimA] > 0) && (dimTypes[dimB] < 0)) {
          useDimA = dimB;
          useDimB = dimA;
        } else {
          useDimA = dimA;
          useDimB = dimB;
        }

        rawPoints = new double[total][2];

        index = 0;
        for (final IMatrix run : loaded) {
          size = run.m();
          for (row = 0; row < size; row++) {
            rawPoints[index][0] = run.getDouble(row, useDimA);
            rawPoints[index][1] = run.getDouble(row, useDimB);
            index++;
          }
        }

        Arrays.sort(rawPoints, this.m_sorter);

        useName = name;
        if (dimTypes.length > 2) {
          useName = ((((((useName + '_')
              + ((dimTypes[useDimA] < 0) ? 't' : 'f')) + useDimA)//
              + '_') + ((dimTypes[useDimB] < 0) ? 't' : 'f')) + useDimB);
        }

        ds = new FittingExampleDataset(useName, //
            new DoubleMatrix2D(rawPoints), //
            DimensionRelationshipModels.getModels((dimTypes[useDimA] < 0),
                (dimTypes[useDimB] < 0)),
            loaded.size());
        synchronized (list) {
          list.add(ds);
        }
      }
    }
  }

  /**
   * The main method
   *
   * @param args
   *          the arguments
   * @throws IOException
   *           if i/o fails
   */
  public static final void main(final String[] args) throws IOException {
    final ArrayListView<FittingExampleDataset> data;
    final Path root;
    final Random rand;
    Path dest;
    double[] randFitting;
    int index;

    data = new FittingExampleDatasets(null, true).call();
    root = PathUtils.getTempDir();
    rand = new Random();
    index = 0;

    for (final FittingExampleDataset ds : data) {

      System.out.print(++index);
      System.out.print('\t');
      System.out.println(ds);

      randFitting = new double[ds.models.get(0).getParameterCount()];
      ds.models.get(0).createParameterGuesser(ds.data)
          .createRandomGuess(randFitting, rand);

      dest = PathUtils.createPathInside(root,
          (Integer.toString(index) + ".txt")); //$NON-NLS-1$
      ds.plot(dest, ds.models.get(0), randFitting, 0d);
    }
  }

  /** The internal sorter class */
  private static final class __Sorter implements Comparator<double[]> {

    /** create */
    __Sorter() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final int compare(final double[] o1, final double[] o2) {
      int idx, cr;
      if (o1 == o2) {
        return 0;
      }

      idx = 0;
      for (final double d : o1) {
        cr = Double.compare(d, o2[idx++]);
        if (cr != 0) {
          return cr;
        }
      }

      return 0;
    }

  }

  /** the internal append resource job */
  private final class __AppendResource implements Callable<Void> {

    /** the resource */
    private final String m_resource;

    /** the list */
    private final ArrayList<FittingExampleDataset> m_list;

    /**
     * create the job
     *
     * @param resource
     *          the resource
     * @param list
     *          the list
     */
    __AppendResource(final String resource,
        final ArrayList<FittingExampleDataset> list) {
      super();
      this.m_list = list;
      this.m_resource = resource;
    }

    /** {@inheritDoc} */
    @Override
    public final Void call() throws Exception {
      if ((FittingExampleDatasets.this.m_logger != null)
          && (FittingExampleDatasets.this.m_logger
              .isLoggable(Level.FINER))) {
        FittingExampleDatasets.this.m_logger
            .finer("Now beginning to load resource " + this.m_resource); //$NON-NLS-1$
      }
      FittingExampleDatasets.this._appendResource(this.m_resource,
          this.m_list);
      if ((FittingExampleDatasets.this.m_logger != null)
          && (FittingExampleDatasets.this.m_logger
              .isLoggable(Level.FINER))) {
        FittingExampleDatasets.this.m_logger
            .finer("Finished loading resource " + this.m_resource); //$NON-NLS-1$
      }
      return null;
    }
  }
}
