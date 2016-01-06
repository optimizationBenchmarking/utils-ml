package examples.org.optimizationBenchmarking.utils.ml.clustering;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.Callable;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;

/** The loader for the clustering example data sets. */
public final class ClusteringExampleDatasets
    implements Callable<ArrayListView<ClusteringExampleDataset>> {

  /** the internal data set list */
  private static final String[] DATASETS = { //
      "simple2", //$NON-NLS-1$
      "simple3", //$NON-NLS-1$
      "simple4", //$NON-NLS-1$
      "exp3-2", //$NON-NLS-1$
      "exp4-2", //$NON-NLS-1$
      "exp5-2", //$NON-NLS-1$
      "exp6-2", //$NON-NLS-1$
      "exp3-10", //$NON-NLS-1$
      "exp4-10", //$NON-NLS-1$
      "exp5-10", //$NON-NLS-1$
      "iris", //$NON-NLS-1$
  };

  /** create */
  public ClusteringExampleDatasets() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final ArrayListView<ClusteringExampleDataset> call()
      throws IOException {
    final ClusteringExampleDataset[] ds;
    int i;

    i = ClusteringExampleDatasets.DATASETS.length;
    ds = new ClusteringExampleDataset[i];
    for (final String name : ClusteringExampleDatasets.DATASETS) {
      ds[--i] = ClusteringExampleDataset._load(name);
    }
    Arrays.sort(ds);
    return new ArrayListView<>(ds);
  }

}
