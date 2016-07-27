package examples.org.optimizationBenchmarking.utils.ml.fitting;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Path;
import java.util.concurrent.Future;
import java.util.logging.Logger;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.io.paths.PathUtils;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.ArithmeticMeanAggregate;
import org.optimizationBenchmarking.utils.math.statistics.aggregate.QuantileAggregate;
import org.optimizationBenchmarking.utils.math.statistics.ranking.ETieStrategy;
import org.optimizationBenchmarking.utils.math.statistics.ranking.RankingStrategy;
import org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls.CMAESLSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.dels.DELSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.esls.ESLSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.lssimplex.LSSimplexFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;
import org.optimizationBenchmarking.utils.parallel.Execute;
import org.optimizationBenchmarking.utils.parsers.LoggerParser;
import org.optimizationBenchmarking.utils.text.ETextCase;
import org.optimizationBenchmarking.utils.text.numbers.SimpleNumberAppender;

/** The examples for fitting. */
public class FittingExamples {

  /** the fitters */
  public static final ArrayListView<IFunctionFitter> FITTERS = //
  new ArrayListView<>(new IFunctionFitter[] { //
      CMAESLSFitter.getInstance(), //
      DELSFitter.getInstance(), //
      ESLSFitter.getInstance(), //
      LSSimplexFitter.getInstance(),//
  }, false);

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
      wait[i] = Execute.parallel(new _FitterJob(logger,
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
  private static final void __print(final FitterOutcome[] outcomes,
      final ArrayListView<MultiFittingExampleDataset> data,
      final Path path) throws IOException {
    final ArithmeticMeanAggregate[] meanQualityRanks, meanRuntimeRanks;
    final QuantileAggregate[] medianQualityRanks, medianRuntimeRanks;
    final double[][] qualities, runtimes;
    FittingOutcome current;
    RankingStrategy ranking;
    int index;

    index = outcomes.length;
    meanQualityRanks = new ArithmeticMeanAggregate[index];
    medianQualityRanks = new QuantileAggregate[index];
    meanRuntimeRanks = new ArithmeticMeanAggregate[index];
    medianRuntimeRanks = new QuantileAggregate[index];
    qualities = new double[index][];
    runtimes = new double[index][];

    for (; (--index) >= 0;) {
      meanQualityRanks[index] = new ArithmeticMeanAggregate();
      meanRuntimeRanks[index] = new ArithmeticMeanAggregate();
      medianQualityRanks[index] = new QuantileAggregate(0.5d);
      medianRuntimeRanks[index] = new QuantileAggregate(0.5d);
    }
    try (final OutputStream os = PathUtils.openOutputStream(
        PathUtils.createPathInside(path, "summary.txt"))) { //$NON-NLS-1$
      try (final OutputStreamWriter osw = new OutputStreamWriter(os)) {
        try (final BufferedWriter bw = new BufferedWriter(osw)) {
          for (final ETieStrategy ties : new ETieStrategy[] {
              ETieStrategy.MINIMUM_TIGHT, ETieStrategy.MINIMUM,
              ETieStrategy.AVERAGE }) {
            ranking = new RankingStrategy(null, ties);
            bw.write("======================= ranking method: ");//$NON-NLS-1$
            bw.write(ties.toString());
            bw.write(" ======================= ");//$NON-NLS-1$
            bw.newLine();
            bw.newLine();
            for (; (--index) >= 0;) {
              meanQualityRanks[index].reset();
              meanRuntimeRanks[index].reset();
              medianQualityRanks[index].reset();
              medianRuntimeRanks[index].reset();
            }

            outer: for (final MultiFittingExampleDataset dataSet : data) {
              for (index = outcomes.length; (--index) >= 0;) {
                current = outcomes[index].outcomeForDataset(dataSet);
                if (current == null) {
                  continue outer;
                }
                qualities[index] = current._getQualities();
                runtimes[index] = current._getRuntimes();
              }

              ranking.rank(qualities, meanQualityRanks);
              ranking.rank(qualities, medianQualityRanks);
              ranking.rank(runtimes, meanRuntimeRanks);
              ranking.rank(runtimes, medianRuntimeRanks);
            }

            bw.write("Mean Quality Ranks");//$NON-NLS-1$
            for (index = 0; index < outcomes.length; index++) {
              bw.newLine();
              bw.write(outcomes[index].fitter.getClass().getSimpleName());
              bw.write(':');
              bw.write('\t');
              bw.write(SimpleNumberAppender.INSTANCE.toString(
                  meanQualityRanks[index].doubleValue(),
                  ETextCase.AT_SENTENCE_START));
            }
            bw.newLine();
            bw.newLine();
            bw.write("Median Quality Ranks");//$NON-NLS-1$
            for (index = 0; index < outcomes.length; index++) {
              bw.newLine();
              bw.write(outcomes[index].fitter.getClass().getSimpleName());
              bw.write(':');
              bw.write('\t');
              bw.write(SimpleNumberAppender.INSTANCE.toString(
                  medianQualityRanks[index].doubleValue(),
                  ETextCase.AT_SENTENCE_START));
            }
            bw.newLine();
            bw.newLine();
            bw.write("Mean Runtime Ranks");//$NON-NLS-1$
            for (index = 0; index < outcomes.length; index++) {
              bw.newLine();
              bw.write(outcomes[index].fitter.getClass().getSimpleName());
              bw.write(':');
              bw.write('\t');
              bw.write(SimpleNumberAppender.INSTANCE.toString(
                  meanRuntimeRanks[index].doubleValue(),
                  ETextCase.AT_SENTENCE_START));
            }
            bw.newLine();
            bw.newLine();
            bw.write("Median Runtime Ranks");//$NON-NLS-1$
            for (index = 0; index < outcomes.length; index++) {
              bw.newLine();
              bw.write(outcomes[index].fitter.getClass().getSimpleName());
              bw.write(':');
              bw.write('\t');
              bw.write(SimpleNumberAppender.INSTANCE.toString(
                  medianRuntimeRanks[index].doubleValue(),
                  ETextCase.AT_SENTENCE_START));
            }
            bw.newLine();
            bw.newLine();
          }
        }
      }
    }
  }
}
