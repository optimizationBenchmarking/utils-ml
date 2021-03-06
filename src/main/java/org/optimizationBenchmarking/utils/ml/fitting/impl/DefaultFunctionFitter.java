package org.optimizationBenchmarking.utils.ml.fitting.impl;

import java.util.ArrayList;

import org.optimizationBenchmarking.utils.collections.iterators.IterableIterator;
import org.optimizationBenchmarking.utils.collections.lists.ArrayListView;
import org.optimizationBenchmarking.utils.error.ErrorUtils;
import org.optimizationBenchmarking.utils.ml.fitting.impl.cmaesls.CMAESLSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.debug.DebugFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.dels.DELSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.impl.esls.ESLSFitter;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IFunctionFitter;

/**
 * The default function fitter.
 */
public final class DefaultFunctionFitter {

  /** the forbidden constructor */
  private DefaultFunctionFitter() {
    ErrorUtils.doNotCall();
  }

  /** No fitter was found. */
  private static final void __noFitter() {
    throw new IllegalStateException("No useable function fitter found."); //$NON-NLS-1$
  }

  /**
   * Get the default function fitter
   *
   * @return the default function fitter
   * @throws IllegalStateException
   *           if no working fitter exists
   */
  public static final IFunctionFitter getInstance() {
    final IFunctionFitter inst;
    inst = __DefaultHolder.INSTANCE;
    if (inst == null) {
      DefaultFunctionFitter.__noFitter();
    }
    return inst;
  }

  /**
   * Get the available function fitter instances
   *
   * @return the available function fitter instances
   * @throws IllegalStateException
   *           if no working fitter exists
   */
  public static final ArrayListView<IFunctionFitter> getAllInstance() {
    final ArrayListView<IFunctionFitter> fitters;
    fitters = __AllHolder.INSTANCES;
    if (fitters == null) {
      DefaultFunctionFitter.__noFitter();
    }
    return fitters;
  }

  /**
   * Set the available function fitters.
   *
   * @param fitters
   *          the fitters
   */
  public static final void setAllInstances(
      final ArrayListView<IFunctionFitter> fitters) {
    if ((fitters == null) || (fitters.isEmpty())) {
      throw new IllegalArgumentException(
          "Default fitters cannot be null or empty."); //$NON-NLS-1$
    }
    __AllHolder.INSTANCES = fitters;
  }

  /** the internal holder for the default fitter */
  private static final class __DefaultHolder {

    /** the fitter */
    static final IFunctionFitter INSTANCE;

    static {
      IFunctionFitter current2;
      find: {
        for (final IFunctionFitter current : new __FitterIterator()) {
          if ((current != null) && (current.canUse())) {
            INSTANCE = current;
            break find;
          }
        }
        current2 = DebugFitter.getInstance();
        if ((current2 != null) && (current2.canUse())) {
          INSTANCE = current2;
          break find;
        }
        INSTANCE = null;
      }
    }
  }

  /** the internal holder for all available fitters */
  private static final class __AllHolder {

    /** the instances */
    static volatile ArrayListView<IFunctionFitter> INSTANCES;

    static {
      ArrayList<IFunctionFitter> fitters;

      fitters = new ArrayList<>();
      for (final IFunctionFitter current : new __FitterIterator()) {
        if ((current != null) && (current.canUse())) {
          fitters.add(current);
        }
      }

      if (!(fitters.isEmpty())) {
        __AllHolder.INSTANCES = ArrayListView.collectionToView(fitters,
            false);
      } else {
        __AllHolder.INSTANCES = null;
      }
    }
  }

  /** the internal iterator for fitters */
  private static final class __FitterIterator
      extends IterableIterator<IFunctionFitter> {
    /** the fitter index */
    private int m_index;

    /** create */
    __FitterIterator() {
      super();
    }

    /** {@inheritDoc} */
    @Override
    public final boolean hasNext() {
      return (this.m_index <= 2);
    }

    /** {@inheritDoc} */
    @Override
    public final IFunctionFitter next() {
      switch (this.m_index++) {
        case 0: {
          return ESLSFitter.getInstance();
        }
        case 1: {
          return DELSFitter.getInstance();
        }
        case 2: {
          return CMAESLSFitter.getInstance();
        }
        default: {
          return super.next();
        }
      }
    }
  }
}
