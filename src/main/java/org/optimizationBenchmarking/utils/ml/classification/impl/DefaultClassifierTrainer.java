package org.optimizationBenchmarking.utils.ml.classification.impl;

import java.util.ArrayList;

import org.optimizationBenchmarking.utils.collections.iterators.IterableIterator;
import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerPruned;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerPrunedBinary;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerReducedErrorPruned;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerReducedErrorPrunedBinary;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerUnpruned;
import org.optimizationBenchmarking.utils.ml.classification.impl.weka.WekaJ48TrainerUnprunedBinary;
import org.optimizationBenchmarking.utils.ml.classification.spec.IClassifierTrainer;

/**
 * The default classifier trainer.
 */
public final class DefaultClassifierTrainer {

  /** the forbidden constructor */
  private DefaultClassifierTrainer() {
    ErrorUtils.doNotCall();
  }

  /** No fitter was found. */
  private static final void __noTrainer() {
    throw new IllegalStateException(
        "No useable classifier trainer found."); //$NON-NLS-1$
  }

  /**
   * Get the default classifier trainer
   *
   * @return the default classifier trainer
   * @throws IllegalStateException
   *           if no working fitter exists
   */
  public static final IClassifierTrainer getInstance() {
    final IClassifierTrainer inst;
    inst = __DefaultHolder.INSTANCE;
    if (inst == null) {
      DefaultClassifierTrainer.__noTrainer();
    }
    return inst;
  }

  /**
   * Get the available classifier trainer instances
   *
   * @return the available classifier trainer instances
   * @throws IllegalStateException
   *           if no working fitter exists
   */
  public static final ArrayListView<IClassifierTrainer> getAllInstance() {
    final ArrayListView<IClassifierTrainer> trainers;
    trainers = __AllHolder.INSTANCES;
    if (trainers == null) {
      DefaultClassifierTrainer.__noTrainer();
    }
    return trainers;
  }

  /** the internal holder for the default fitter */
  private static final class __DefaultHolder {

    /** the fitter */
    static final IClassifierTrainer INSTANCE;

    static {
      find: {
        for (final IClassifierTrainer current : new __TrainerIterator()) {
          if ((current != null) && (current.canUse())) {
            INSTANCE = current;
            break find;
          }
        }
        INSTANCE = null;
      }
    }
  }

  /** the internal holder for all available trainers */
  private static final class __AllHolder {

    /** the instances */
    static final ArrayListView<IClassifierTrainer> INSTANCES;

    static {
      ArrayList<IClassifierTrainer> trainers;

      trainers = new ArrayList<>();
      for (final IClassifierTrainer current : new __TrainerIterator()) {
        if ((current != null) && (current.canUse())) {
          trainers.add(current);
        }
      }

      if (!(trainers.isEmpty())) {
        INSTANCES = ArrayListView.collectionToView(trainers, false);
      } else {
        INSTANCES = null;
      }
    }
  }

  /** the internal iterator for trainers */
  private static final class __TrainerIterator
      extends IterableIterator<IClassifierTrainer> {
    /** the fitter index */
    private int m_index;

    /** create */
    __TrainerIterator() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final boolean hasNext() {
      return (this.m_index <= 5);
    }

    /** {@inheritDoc} */
    @Override
    public final IClassifierTrainer next() {
      switch (this.m_index++) {
        case 0: {
          return WekaJ48TrainerPruned.getInstance();
        }
        case 1: {
          return WekaJ48TrainerPrunedBinary.getInstance();
        }
        case 2: {
          return WekaJ48TrainerReducedErrorPruned.getInstance();
        }
        case 3: {
          return WekaJ48TrainerReducedErrorPrunedBinary.getInstance();
        }
        case 4: {
          return WekaJ48TrainerUnpruned.getInstance();
        }
        case 5: {
          return WekaJ48TrainerUnprunedBinary.getInstance();
        }
        default: {
          return super.next();
        }
      }
    }
  }
}
