package org.optimizationBenchmarking.utils.ml.clustering.impl;

import java.util.LinkedHashSet;

import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDataClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.impl.Rbased.RBasedDistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDataClusteringJobBuilder;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusterer;
import org.optimizationBenchmarking.utils.ml.clustering.spec.IDistanceClusteringJobBuilder;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/** The default clusterers. */
public final class DefaultClusterer {

  /** the forbidden constructor */
  private DefaultClusterer() {
    ErrorUtils.doNotCall();
  }

  /**
   * Get the default data clusterer
   *
   * @return the default data clusterer
   */
  public static final IDataClusterer getDataInstance() {
    return __DefaultDataClusterer.INSTANCE;
  }

  /**
   * Get all available data clusterers
   *
   * @return all available data clusterers
   */
  public static final ArrayListView<IDataClusterer> getAllDataInstances() {
    return __AllDataClusterers.INSTANCES;
  }

  /**
   * Get the default distance clusterer
   *
   * @return the default distance clusterer
   */
  public static final IDistanceClusterer getDistanceInstance() {
    return __DefaultDistanceClusterer.INSTANCE;
  }

  /**
   * Get all available distance clusterers
   *
   * @return all available distance clusterers
   */
  public static final ArrayListView<IDistanceClusterer> getAllDistanceInstances() {
    return __AllDistanceClusterers.INSTANCES;
  }

  /** the default data clusterer */
  private static final class __DefaultDataClusterer
      implements IDataClusterer {
    /** the instance */
    static final IDataClusterer INSTANCE;

    static {
      IDataClusterer inst;
      inst = RBasedDistanceClusterer.getInstance();
      if (inst.canUse()) {
        INSTANCE = inst;
      } else {
        inst = RBasedDataClusterer.getInstance();
        if (inst.canUse()) {
          INSTANCE = inst;
        } else {
          INSTANCE = new __DefaultDataClusterer();
        }
      }
    }

    /** create */
    private __DefaultDataClusterer() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final boolean canUse() {
      return false;
    }

    /** {@inheritDoc} */
    @Override
    public final void checkCanUse() {
      throw __DefaultDataClusterer.__noDataInstance();
    }

    /** {@inheritDoc} */
    @Override
    public final String toString() {
      return "Unusable Data Clusterer"; //$NON-NLS-1$
    }

    /** {@inheritDoc} */
    @Override
    public final void toText(final ITextOutput textOut) {
      textOut.append(this.toString());
    }

    @Override
    public IDataClusteringJobBuilder use() {
      throw __DefaultDataClusterer.__noDataInstance();
    }

    /**
     * No useable data clusterer was found.
     *
     * @return the exception
     */
    private static final IllegalStateException __noDataInstance() {
      throw new IllegalStateException(
          "No useable data clusterer detected."); //$NON-NLS-1$
    }
  }

  /** the all data clusterer */
  private static final class __AllDataClusterers {
    /** all data clusterer instnaces */
    static final ArrayListView<IDataClusterer> INSTANCES;

    static {
      LinkedHashSet<IDataClusterer> insts;
      IDataClusterer inst;

      inst = __DefaultDataClusterer.INSTANCE;
      if (inst instanceof __DefaultDataClusterer) {
        INSTANCES = new ArrayListView<>(new IDataClusterer[] { inst },
            false);
      } else {
        insts = new LinkedHashSet<>();
        insts.add(inst);
        inst = RBasedDistanceClusterer.getInstance();
        if (inst.canUse()) {
          insts.add(inst);
        }
        inst = RBasedDataClusterer.getInstance();
        if (inst.canUse()) {
          insts.add(inst);
        }
        INSTANCES = ArrayListView.collectionToView(insts);
      }
    }
  }

  /** the default distance clusterer */
  private static final class __DefaultDistanceClusterer
      implements IDistanceClusterer {
    /** the instance */
    static final IDistanceClusterer INSTANCE;

    static {
      IDistanceClusterer inst;
      inst = RBasedDistanceClusterer.getInstance();
      if (inst.canUse()) {
        INSTANCE = inst;
      } else {
        INSTANCE = new __DefaultDistanceClusterer();
      }
    }

    /** create */
    private __DefaultDistanceClusterer() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final boolean canUse() {
      return false;
    }

    /** {@inheritDoc} */
    @Override
    public final void checkCanUse() {
      throw __DefaultDistanceClusterer.__noDistanceInstance();
    }

    /** {@inheritDoc} */
    @Override
    public final String toString() {
      return "Unusable Distance Clusterer"; //$NON-NLS-1$
    }

    /** {@inheritDoc} */
    @Override
    public final void toText(final ITextOutput textOut) {
      textOut.append(this.toString());
    }

    @Override
    public IDistanceClusteringJobBuilder use() {
      throw __DefaultDistanceClusterer.__noDistanceInstance();
    }

    /**
     * No useable distance clusterer was found.
     *
     * @return the exception
     */
    private static final IllegalStateException __noDistanceInstance() {
      throw new IllegalStateException(
          "No useable distance clusterer detected."); //$NON-NLS-1$
    }
  }

  /** the all distance clusterer */
  private static final class __AllDistanceClusterers {
    /** all distance clusterer instnaces */
    static final ArrayListView<IDistanceClusterer> INSTANCES;

    static {
      LinkedHashSet<IDistanceClusterer> insts;
      IDistanceClusterer inst;

      inst = __DefaultDistanceClusterer.INSTANCE;
      if (inst instanceof __DefaultDistanceClusterer) {
        INSTANCES = new ArrayListView<>(new IDistanceClusterer[] { inst },
            false);
      } else {
        insts = new LinkedHashSet<>();
        insts.add(inst);
        inst = RBasedDistanceClusterer.getInstance();
        if (inst.canUse()) {
          insts.add(inst);
        }
        INSTANCES = ArrayListView.collectionToView(insts);
      }
    }
  }
}