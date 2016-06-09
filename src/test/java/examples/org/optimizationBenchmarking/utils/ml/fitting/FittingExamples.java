package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.Future;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.ml.fitting.impl.dels.DELSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex.LSSimplexFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;
import org.optimizationBenchmarking.utils.parallel.Execute;
import org.optimizationBenchmarking.utils.parsers.LoggerParser;

/** The examples for fitting. */
public class FittingExamples {

  /** the fitters */
  public static final ArrayListView<IFunctionFitter> FITTERS = //
  new ArrayListView<>(new IFunctionFitter[] { //
      DELSFitter.getInstance(), //
      LSSimplexFitter.getInstance(),//
  });

  /**
   * The main entry point
   *
   * @param args
   *          the arguments: ignored
   * @throws Exception
   *           if something fails
   */
  @SuppressWarnings({ "rawtypes", "unchecked" })
  public static final void main(final String[] args) throws Exception {
    final ArrayListView<MultiFittingExampleDataset> data;
    final Path dest;
    final Future[] wait;
    final Logger logger;
    final FitterOutcome[] res;
    int i;

    logger = LoggerParser.INSTANCE.parseString("global;ALL");//$NON-NLS-1$

    data = MultiFittingExampleDatasets.EXAMPLES;

    dest = PathUtils.createPathInside(PathUtils.getTempDir(), "results"); //$NON-NLS-1$
    i = FittingExamples.FITTERS.size();
    wait = new Future[i];
    for (; (--i) >= 0;) {
      wait[i] = Execute.submitToCommonPool(new _FitterJob(logger,
          FittingExamples.FITTERS.get(i), data, dest));
    }

    res = new FitterOutcome[wait.length];
    Execute.join(wait, res, 0, false);
    FittingExamples.__print(res, data, dest);
  }

  /**
   * Print the fitting outcome
   *
   * @param outcomes
   *          the fitting outcomes
   * @param data
   *          the data sets
   * @param path
   *          the paths
   * @throws IOException
   *           if i/o fails
   */
  @SuppressWarnings("unchecked")
  private static final void __print(final FitterOutcome[] outcomes,
      final ArrayListView<MultiFittingExampleDataset> data,
      final Path path) throws IOException {
    HashMap<MultiFittingExampleDataset, ArrayListView<SingleFittingOutcome>>[] results;
    long[] totalRanks;
    int[] lastRanks;
    ArrayListView<SingleFittingOutcome> current;
    ArrayList<SingleFittingOutcome> list;
    int index, nextRank, prevRank, counter;
    double lastQuality;

    index = outcomes.length;
    totalRanks = new long[index];
    lastRanks = new int[index];
    results = new HashMap[index];

    for (; (--index) >= 0;) {
      results[index] = new HashMap<>();
      for (final FittingOutcome outcome : outcomes[index].outcomes) {
        results[index].put(outcome.example, outcome.outcomes);
      }
    }

    list = new ArrayList<>();
    outer: for (final MultiFittingExampleDataset dataSet : data) {
      list.clear();
      for (final HashMap<MultiFittingExampleDataset, ArrayListView<SingleFittingOutcome>> map : results) {
        current = map.get(dataSet);
        if (current == null) {
          continue outer;
        }
        list.addAll(current);
      }
      Collections.sort(list);
      Arrays.fill(lastRanks, 0);
      prevRank = (-1);
      nextRank = counter = 0;
      lastQuality = Double.NEGATIVE_INFINITY;
      looper: for (final SingleFittingOutcome outcome : list) {
        ++counter;
        if (outcome.errors.quality > lastQuality) {
          nextRank = counter;
          lastQuality = outcome.errors.quality;
        }

        index = (-1);
        for (final HashMap<MultiFittingExampleDataset, ArrayListView<SingleFittingOutcome>> map : results) {
          ++index;
          if (map.get(dataSet).contains(outcome)) {
            if (lastRanks[index] < prevRank) {
              totalRanks[index] += nextRank;
            }
            prevRank = nextRank;
            lastRanks[index] = nextRank;
            continue looper;
          }
        }
        throw new IllegalStateException("huh?"); //$NON-NLS-1$
      }
    }

    try (final OutputStream os = PathUtils.openOutputStream(
        PathUtils.createPathInside(path, "summary.txt"))) { //$NON-NLS-1$
      try (final OutputStreamWriter osw = new OutputStreamWriter(os)) {
        try (final BufferedWriter bw = new BufferedWriter(osw)) {
          for (index = 0; index < outcomes.length; index++) {
            bw.write(outcomes[index].fitter.getClass().getSimpleName());
            bw.write(':');
            bw.write('\t');
            bw.write(Long.toString(totalRanks[index]));
            bw.newLine();
          }
        }
      }
    }
  }
}
