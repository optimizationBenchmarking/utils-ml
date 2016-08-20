package org.optimizationBenchmarking.utils.ml.fitting.models;

import java.util.Random;

import org.optimizationBenchmarking.utils.document.spec.IMath;
import org.optimizationBenchmarking.utils.document.spec.IMathRenderable;
import org.optimizationBenchmarking.utils.document.spec.IParameterRenderer;
import org.optimizationBenchmarking.utils.document.spec.IText;
import org.optimizationBenchmarking.utils.math.MathUtils;
import org.optimizationBenchmarking.utils.math.matrix.IMatrix;
import org.optimizationBenchmarking.utils.math.text.INegatableParameterRenderer;
import org.optimizationBenchmarking.utils.ml.fitting.impl.guessers.SamplingBasedParameterGuesser;
import org.optimizationBenchmarking.utils.ml.fitting.spec.IParameterGuesser;
import org.optimizationBenchmarking.utils.text.textOutput.ITextOutput;

/**
 * <p>
 * A model function which may be suitable to model how time-objective value
 * relationships of optimization processes behave if the finally obtained
 * objective value is not {@code 0}: {@code a+b/(1+c*x^d)}.
 * </p>
 * <h2>Alternate Forms</h2>
 * <ol>
 * <li>{@code b/(1+c*x^d) + a}</li>
 * <li>{@code (a*c*x^d+a+b)/(c*x^d+1)}</li>
 * <li>{@code (a*(c*x^d+1)+b)/(c*x^d+1)}</li>
 * </ol>
 * <h2>Derivatives</h2>
 * <p>
 * The derivatives have been obtained with http://www.numberempire.com/.
 * </p>
 * <ol>
 * <li>Original function: {@code a+b/(1+c*x^d)}</li>
 * <li>{@code d/da}: {@code 1}</li>
 * <li>{@code d/db}: {@code 1/(1+c*x^d)}</li>
 * <li>{@code d/dc}: {@code -(         b*x^d)/(1+2*c*x^d+c^2*x^(2*d))}</li>
 * <li>{@code d/dd}: {@code -(c*log(x)*b*x^d)/(1+2*c*x^d+c^2*x^(2*d))}</li>
 * </ol>
 * <h2>Resolution</h2> The resolutions have been obtained with
 * http://www.numberempire.com/ and http://wolframalpha.com/.
 * <h3>One Known Point</h3>
 * <dl>
 * <dt>a.1.1</dt>
 * <dd>{@code a=((1+c*x^d)*y-b)/(1+c*x^d)}</dd>
 * <dt>b.1.1</dt>
 * <dd>{@code b=(y-a)*(c*x^d+1)}</dd>
 * <dt>b.1.2 [=b.1.1]</dt>
 * <dd>{@code -a*c*x^d-a+y*c*x^d+y}</dd>
 * <dt>c.1.1</dt>
 * <dd>{@code c=(a+b-y)/(x^d*(y-a))}</dd>
 * <dt>c.1.2 [=c.1.1]</dt>
 * <dd>{@code c=(x^(-d)*(a+b-y))/(y-a)}</dd>
 * <dt>d.1.1</dt>
 * <dd>{@code d=log((y-a-b)/(c*(a-y)))/log(x)}</dd>
 * </dl>
 * <h3>Two Known Points</h3>
 * <dl>
 * <dt>a.2.1 [from b.1.1]</dt>
 * <dd>{@code a=((1+c*x2^d)*y2-(1+c*x1^d)*y1)/(c*(x2^d-x1^d))}</dd>
 * <dt>a.2.2 [= a.2.1; from b.1.1]</dt>
 * <dd>{@code a=(-c*y1*x1^d+c*y2*x2^d-y1+y2)/(c*(x2^d-x1^d))}</dd>
 * <dt>a.2.3 [from c.1.1]</dt>
 * <dd>
 * {@code a=(-sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))+b*x1^d-b*x2^d-y1*x1^d-y2*x1^d+y1*x2^d+y2*x2^d)/(2*(x2^d-x1^d))}
 * </dd>
 * <dt>a.2.4 [from c.1.1]</dt>
 * <dd>
 * {@code a=(sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))+b*x1^d-b*x2^d-y1*x1^d-y2*x1^d+y1*x2^d+y2*x2^d)/(2*(x2^d-x1^d))}
 * </dd>
 * <dt>a.2.5 [from c.1.1; probably=a.2.4]</dt>
 * <dd>
 * {@code a=(sqrt((x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y2^2+((-2*b*x1^(2*d))+2*b*x2^(2*d)+((-2*x2^(2*d))+4*x1^d*x2^d-2*x1^(2*d))*y1)*y2+(x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y1^2+(2*b*x1^(2*d)-2*b*x2^(2*d))*y1+b^2*x2^(2*d)-2*b^2*x1^d*x2^d+b^2*x1^(2*d))+(x2^d-x1^d)*y2+(x2^d-x1^d)*y1-b*x2^d+b*x1^d)/(2*x2^d-2*x1^d)}
 * </dd>
 * <dt>a.2.6 [from c.1.1; probably=a.2.3]</dt>
 * <dd>
 * {@code a=-(sqrt((x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y2^2+((-2*b*x1^(2*d))+2*b*x2^(2*d)+((-2*x2^(2*d))+4*x1^d*x2^d-2*x1^(2*d))*y1)*y2+(x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y1^2+(2*b*x1^(2*d)-2*b*x2^(2*d))*y1+b^2*x2^(2*d)-2*b^2*x1^d*x2^d+b^2*x1^(2*d))+(x1^d-x2^d)*y2+(x1^d-x2^d)*y1+b*x2^d-b*x1^d)/(2*x2^d-2*x1^d)}
 * </dd>
 * <dt>b.2.1 [from a.1.1]</dt>
 * <dd>
 * {@code b=-((1+c*x1^d+(c^2*x1^d+c)*x2^d)*y2+((-1)-c*x1^d+((-c^2*x1^d)-c)*x2^d)*y1)/(c*x2^d-c*x1^d)}
 * </dd>
 * <dt>b.2.2 [=b.2.1; from a.1.1]</dt>
 * <dd>{@code b=((y1-y2)*(c*x1^d+1)*(c*x2^d+1))/(c*(x2^d-x1^d))}</dd>
 * <dt>b.2.3 [from c.1.1]</dt>
 * <dd>
 * {@code b=((y1-a)*(a-y2)*(x1^d-x2^d))/(a*(x1^d-x2^d)-y1*x1^d+y2*x2^d)}
 * </dd>
 * <dt>b.2.4 [=b.2.3; from c.1.1]</dt>
 * <dd>
 * {@code b=((a*x1^d-a*x2^d+(x2^d-x1^d)*y1)*y2+(a*x1^d-a*x2^d)*y1+a^2*x2^d-a^2*x1^d)/(a*x1^d-a*x2^d-x1^d*y1+x2^d*y2)}
 * </dd>
 * <dt>c.2.1 [from a.1.1]</dt>
 * <dd>
 * {@code c=(x1^(-d)*x2^(-d)*(-sqrt((b*x1^d-b*x2^d+y1*x1^d-y2*x1^d+y1*x2^d-y2*x2^d)^2-4*(y1-y2)*(y1*x1^d*x2^d-y2*x1^d*x2^d))-b*x1^d+b*x2^d-y1*x1^d+y2*x1^d-y1*x2^d+y2*x2^d))/(2*(y1-y2))}
 * </dd>
 * <dt>c.2.2 [from a.1.1]</dt>
 * <dd>
 * {@code c=(x1^(-d)*x2^(-d)*(sqrt((b*x1^d-b*x2^d+y1*x1^d-y2*x1^d+y1*x2^d-y2*x2^d)^2-4*(y1-y2)*(y1*x1^d*x2^d-y2*x1^d*x2^d))-b*x1^d+b*x2^d-y1*x1^d+y2*x1^d-y1*x2^d+y2*x2^d))/(2*(y1-y2))}
 * </dd>
 * <dt>c.2.3 [from a.1.1; probably=c.2.1]</dt>
 * <dd>
 * {@code c=-(sqrt((x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y2^2+((-2*b*x1^(2*d))+2*b*x2^(2*d)+((-2*x2^(2*d))+4*x1^d*x2^d-2*x1^(2*d))*y1)*y2+(x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y1^2+(2*b*x1^(2*d)-2*b*x2^(2*d))*y1+b^2*x2^(2*d)-2*b^2*x1^d*x2^d+b^2*x1^(2*d))+(x1^d+x2^d)*y2+((-x1^d)-x2^d)*y1+b*x2^d-b*x1^d)/(2*x1^d*x2^d*y2-2*x1^d*x2^d*y1)}
 * </dd>
 * <dt>c.2.4 [from a.1.1; probably=c.2.2]</dt>
 * <dd>
 * {@code c=(sqrt((x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y2^2+((-2*b*x1^(2*d))+2*b*x2^(2*d)+((-2*x2^(2*d))+4*x1^d*x2^d-2*x1^(2*d))*y1)*y2+(x1^(2*d)-2*x1^d*x2^d+x2^(2*d))*y1^2+(2*b*x1^(2*d)-2*b*x2^(2*d))*y1+b^2*x2^(2*d)-2*b^2*x1^d*x2^d+b^2*x1^(2*d))+((-x1^d)-x2^d)*y2+(x1^d+x2^d)*y1-b*x2^d+b*x1^d)/(2*x1^d*x2^d*y2-2*x1^d*x2^d*y1)}
 * </dd>
 * <dt>c.2.5 [from b.1.1]</dt>
 * <dd>{@code c=(y1-y2)/(a*x1^d-a*x2^d-x1^d*y1+x2^d*y2)}</dd>
 * <dt>d.2.1 [from c.1.1]</dt>
 * <dd>
 * {@code d=(log(((a-y2)*(a+b-y1))/((a-y1)*(a+b-y2))))/(log(x1)-log(x2))}
 * </dd>
 * </dl>
 * <h3>Three Known Points</h3>
 * <dl>
 * <dt>a.3.1 [from b.2.3 or c.2.5]</dt>
 * <dd>
 * {@code a=-(((x1^d-x3^d)*y1+(x3^d-x2^d)*y2)*y3+(x2^d-x1^d)*y1*y2)/((x3^d-x2^d)*y1+(x1^d-x3^d)*y2+(x2^d-x1^d)*y3)}
 * </dd>
 * <dt>a.3.2 [from b.2.3 or c.2.5; probably=a.3.1]</dt>
 * <dd>
 * {@code a=(y1*y2*x1^d-y1*y3*x1^d-y1*y2*x2^d+y2*y3*x2^d+y1*y3*x3^d-y2*y3*x3^d)/(y2*x1^d-y3*x1^d-y1*x2^d+y3*x2^d+y1*x3^d-y2*x3^d)}
 * </dd>
 * <dt>c.3.1 [from a.2.1 or b.2.2]</dt>
 * <dd>
 * {@code c=-((x2^d-x1^d)*y3+(x1^d-x3^d)*y2+(x3^d-x2^d)*y1)/((x1^d*x3^d-x1^d*x2^d)*y1+(x1^d*x2^d-x2^d*x3^d)*y2+(x2^d-x1^d)*x3^d*y3)}
 * </dd>
 * </dl>
 * <h3>Two Known Points Direct</h3>
 * <dl>
 * <dt>a.d3.1</dt>
 * <dd>
 * {@code a=-(((x1^d-x3^d)*y1+(x3^d-x2^d)*y2)*y3+(x2^d-x1^d)*y1*y2)/((x3^d-x2^d)*y1+(x1^d-x3^d)*y2+(x2^d-x1^d)*y3)}
 * </dd>
 * <dt>a.d3.1</dt>
 * <dd>
 * {@code a=(-y1*y2*x1^d+y1*y3*x1^d+y1*y2*x2^d-y2*y3*x2^d)/(-y2*x1^d+y3*x1^d+y1*x2^d-y3*x2^d)}
 * </dd>
 * <dt>b.d2.1</dt>
 * <dd>
 * {@code b=-((a-y2)*(a*x1^((log(x2^d))/(log(x2)))-a*x2^d-y1*x1^((log(x2^d))/(log(x2)))+y1*x2^d))/(a*x1^((log(x2^d))/(log(x2)))-a*x2^d-y1*x1^((log(x2^d))/(log(x2)))+y2*x2^d)}
 * </dd>
 * <dt>b.d2.2</dt>
 * <dd>
 * {@code b=-((y1-y2)*(c*x2^d+1)*(c*x1^((log(x2^d))/(log(x2)))+1))/(c*(x1^((log(x2^d))/(log(x2)))-x2^d))}
 * </dd>
 * <dt>c.d2.1</dt>
 * <dd>
 * {@code c=(y2-y1)/(-a*x1^((log(x2^d))/(log(x2)))+a*x2^d+y1*x1^((log(x2^d))/(log(x2)))-y2*x2^d)}
 * </dd>
 * <dt>c.d2.2</dt>
 * <dd>
 * {@code c=(x2^(-d)*(-sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))-b*x1^d+b*x2^d-y1*x1^d+y2*x1^d+y1*x2^d-y2*x2^d))/(sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))-b*x1^d+b*x2^d+y1*x1^d-y2*x1^d-y1*x2^d+y2*x2^d)}
 * </dd>
 * <dt>c.d2.3</dt>
 * <dd>
 * {@code c=(x2^(-d)*(sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))-b*x1^d+b*x2^d-y1*x1^d+y2*x1^d+y1*x2^d-y2*x2^d))/(-sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))-b*x1^d+b*x2^d+y1*x1^d-y2*x1^d-y1*x2^d+y2*x2^d)}
 * </dd>
 * <dt>a.d2.1</dt>
 * <dd>
 * {@code a=(c*y1*x1^((log(x2^d))/(log(x2)))-c*y2*x2^d+y1-y2)/(c*(x1^((log(x2^d))/(log(x2)))-x2^d))}
 * </dd>
 * <dt>a.d2.2</dt>
 * <dd>
 * {@code a=(-sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))+b*x1^d-b*x2^d-y1*x1^d-y2*x1^d+y1*x2^d+y2*x2^d)/(2*(x2^d-x1^d))}
 * </dd>
 * <dt>a.d2.3</dt>
 * <dd>
 * {@code a=(sqrt((-b*x1^d+b*x2^d+y1*x1^d+y2*x1^d-y1*x2^d-y2*x2^d)^2-4*(x2^d-x1^d)*(b*y1*x1^d-b*y2*x2^d-y1*y2*x1^d+y1*y2*x2^d))+b*x1^d-b*x2^d-y1*x1^d-y2*x1^d+y1*x2^d+y2*x2^d)/(2*(x2^d-x1^d))}
 * </dd>
 * </dl>
 * <h3>Three Known Points Direct</h3>
 * <dl>
 * <dt>b.d3.1</dt>
 * <dd>
 * {@code b=((((x1^d-x2^d)*x3^(2*d)+(x2^(2*d)-x1^(2*d))*x3^d-x1^d*x2^(2*d)+x1^(2*d)*x2^d)*y1+((x2^d-x1^d)*x3^(2*d)+(x1^(2*d)-x2^(2*d))*x3^d+x1^d*x2^(2*d)-x1^(2*d)*x2^d)*y2)*y3^2+(((x2^d-x1^d)*x3^(2*d)+(x1^(2*d)-x2^(2*d))*x3^d+x1^d*x2^(2*d)-x1^(2*d)*x2^d)*y1^2+((x1^d-x2^d)*x3^(2*d)+(x2^(2*d)-x1^(2*d))*x3^d-x1^d*x2^(2*d)+x1^(2*d)*x2^d)*y2^2)*y3+((-x1^(2*d)*x2^d)+x1^d*x2^(2*d)+(x1^(2*d)-x2^(2*d))*x3^d+(x2^d-x1^d)*x3^(2*d))*y1*y2^2+(x1^(2*d)*x2^d-x1^d*x2^(2*d)+(x2^(2*d)-x1^(2*d))*x3^d+(x1^d-x2^d)*x3^(2*d))*y1^2*y2)/((x1^d*x3^(2*d)-2*x1^d*x2^d*x3^d+x1^d*x2^(2*d))*y1^2+(((-x1^d)-x2^d)*x3^(2*d)+(x1^(2*d)+2*x1^d*x2^d+x2^(2*d))*x3^d-x1^d*x2^(2*d)-x1^(2*d)*x2^d)*y1*y2+(x2^d*x3^(2*d)-2*x1^d*x2^d*x3^d+x1^(2*d)*x2^d)*y2^2+(((-x1^(2*d)*x2^d)+x1^d*x2^(2*d)+((-x2^(2*d))+2*x1^d*x2^d-x1^(2*d))*x3^d+(x1^d-x2^d)*x3^(2*d))*y2+(x1^(2*d)*x2^d-x1^d*x2^(2*d)+((-x2^(2*d))+2*x1^d*x2^d-x1^(2*d))*x3^d+(x2^d-x1^d)*x3^(2*d))*y1)*y3+(x2^(2*d)-2*x1^d*x2^d+x1^(2*d))*x3^d*y3^2)}
 * </dd>
 * <dt>b.d3.2</dt>
 * <dd>
 * {@code b=(-y1*y2*x1^d+y1*y3*x1^d+y2*y3*x1^d-y3^2*x1^d+y1*y2*x2^d-y1*y3*x2^d-y2*y3*x2^d+y3^2*x2^d)/(y2*x1^d-y3*x1^d-y1*x2^d+y3*x2^d)}
 * </dd>
 * <dt>c.d3.1</dt>
 * <dd>
 * {@code c=-((x2^d-x1^d)*y3+(x1^d-x3^d)*y2+(x3^d-x2^d)*y1)/((x1^d*x3^d-x1^d*x2^d)*y1+(x1^d*x2^d-x2^d*x3^d)*y2+(x2^d-x1^d)*x3^d*y3)}
 * </dd>
 * <dt>c.d3.2</dt>
 * <dd>
 * {@code c=(x1^(-d)*x2^(-d)*(-y2*x1^d+y3*x1^d+y1*x2^d-y3*x2^d))/(y2-y1)}
 * </dd>
 * </dl>
 * _ModelBase
 */
public final class LogisticModelWithOffsetOverLogX extends _ModelBase {

  /** create */
  public LogisticModelWithOffsetOverLogX() {
    super();
  }

  /** {@inheritDoc} */
  @Override
  public final String toString() {
    return "generalized logistic model"; //$NON-NLS-1$
  }

  /** {@inheritDoc} */
  @Override
  public final double value(final double x, final double[] parameters) {
    double res, a;

    res = 1d + (_ModelBase._pow(x, parameters[3]) * parameters[2]);
    if (MathUtils.isFinite(res)) {
      res = (parameters[1] / res) + (a = parameters[0]);
      return ((MathUtils.isFinite(res)) ? res
          : (MathUtils.isFinite(a) ? a : 0d));
    }
    return parameters[0];
  }

  /** {@inheritDoc} */
  @Override
  public final void gradient(final double x, final double[] parameters,
      final double[] gradient) {
    final double b, c, d, xd, cxd, bxd, div;

    gradient[0] = 1;

    d = parameters[3];
    xd = _ModelBase._pow(x, d);

    if ((xd == 0d) || ((cxd = ((c = parameters[2]) * xd)) <= 0d)) {
      gradient[1] = 1d;
      gradient[2] = gradient[3] = 0d;
      return;
    }

    b = parameters[1];
    gradient[1] = _ModelBase._gradient((1d / (1d + cxd)), b);

    bxd = (b * xd);
    if (bxd == 0d) {
      gradient[2] = gradient[3] = 0d;
      return;
    }

    div = _ModelBase._add(1d, 2d * cxd, cxd * cxd);
    gradient[2] = _ModelBase._gradient(((-bxd) / div), c);
    gradient[3] = _ModelBase
        ._gradient(((-(c * bxd * _ModelBase._log(x))) / div), d);
  }

  /** {@inheritDoc} */
  @Override
  public final int getParameterCount() {
    return 4;
  }

  /** {@inheritDoc} */
  @Override
  public final void mathRender(final ITextOutput out,
      final IParameterRenderer renderer, final IMathRenderable x) {
    renderer.renderParameter(0, out);
    out.append('+');
    renderer.renderParameter(1, out);
    out.append('/');
    out.append('(');
    out.append('1');
    out.append('+');
    renderer.renderParameter(2, out);
    out.append('*');
    x.mathRender(out, renderer);
    out.append('^');
    renderer.renderParameter(3, out);
    out.append(')');
  }

  /** {@inheritDoc} */
  @SuppressWarnings({ "resource", "null" })
  @Override
  public final void mathRender(final IMath out,
      final IParameterRenderer renderer, final IMathRenderable x) {

    final INegatableParameterRenderer negatableRenderer;
    final IMath closerB, closerC;
    final boolean negateB, negateC;

    if (renderer instanceof INegatableParameterRenderer) {
      negatableRenderer = ((INegatableParameterRenderer) (renderer));
    } else {
      negatableRenderer = null;
    }

    if ((negatableRenderer != null) && (negatableRenderer.isNegative(1))) {
      negateB = true;
      closerB = out.sub();
    } else {
      negateB = false;
      closerB = out.add();
    }
    try {

      renderer.renderParameter(0, closerB);

      try (final IMath div = closerB.div()) {
        if (negateB) {
          negatableRenderer.renderNegatedParameter(1, div);
        } else {
          renderer.renderParameter(1, div);
        }

        if ((negatableRenderer != null)
            && (negatableRenderer.isNegative(2))) {
          negateC = true;
          closerC = div.sub();
        } else {
          negateC = false;
          closerC = div.add();
        }

        try {
          try (final IText num = closerC.number()) {
            num.append('1');
          }
          try (final IMath mul = closerC.mul()) {

            if (negateC) {
              negatableRenderer.renderNegatedParameter(2, mul);
            } else {
              renderer.renderParameter(2, mul);
            }

            try (final IMath pow = mul.pow()) {
              x.mathRender(pow, renderer);
              renderer.renderParameter(3, pow);
            }
          }

        } finally {
          closerC.close();
        }
      }
    } finally {
      closerB.close();
    }
  }

  /** {@inheritDoc} */
  @Override
  public IParameterGuesser createParameterGuesser(final IMatrix data) {
    return new __LogisticModelWithOffsetOverLogXParameterGuesser(data);
  }

  /**
   * Compute {@code a} from one point and {@code b}, {@code c}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code a}
   */
  static final double _a_x1y1bcd(final double x1, final double y1,
      final double b, final double c, final double d) {
    final double cxd;
    cxd = c * _ModelBase._pow(x1, d);
    return (((1d + cxd) * y1) - b) / (1d + cxd);
  }

  /**
   * Compute {@code b} from one point and {@code a}, {@code c}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param c
   *          the value of {@code c}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code b}
   */
  static final double _b_x1y1acd(final double x1, final double y1,
      final double a, final double c, final double d) {
    return (y1 - a) * ((c * _ModelBase._pow(x1, d)) + 1d);
  }

  /**
   * Compute {@code c} from one point and {@code a}, {@code b}, and
   * {@code d}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param d
   *          the value of {@code d}
   * @return the value of {@code c}
   */
  static final double _c_x1y1abd(final double x1, final double y1,
      final double a, final double b, final double d) {
    return ((a + b) - y1) / (_ModelBase._pow(x1, d) * (y1 - a));
  }

  /**
   * Compute {@code d} from one point and {@code a}, {@code b}, and
   * {@code c}.
   *
   * @param x1
   *          the point's {@code x}-coordinate
   * @param y1
   *          the point's {@code y}-coordinate
   * @param a
   *          the value of {@code a}
   * @param b
   *          the value of {@code b}
   * @param c
   *          the value of {@code c}
   * @return the value of {@code d}
   */
  static final double _d_x1y1abc(final double x1, final double y1,
      final double a, final double b, final double c) {
    return _ModelBase._log((y1 - a - b) / (c * (a - y1)))
        / _ModelBase._log(x1);
  }

  /**
   * the internal fallback routine
   *
   * @param minY
   *          the minimal y coordinate
   * @param maxY
   *          the maximum y coordinate
   * @param dest
   *          the destination array
   * @param random
   *          the random number generator
   */
  static final void _fallback(final double minY, final double maxY,
      final double[] dest, final Random random) {
    double temp;
    int steps;

    if (random.nextBoolean()) {
      dest[0] = minY;

      steps = 100;
      do {
        dest[1] = temp = (maxY - minY)
            * (1d + (0.1d * random.nextGaussian()));
      } while ((((--steps) > 0) && (temp < 1e-13d))
          || (!(MathUtils.isFinite(temp))));

      steps = 100;
      do {
        dest[2] = temp = _ModelBase._exp(-12d * random.nextDouble());
      } while ((((--steps) > 0) && (temp < 1e-13d))
          || (!(MathUtils.isFinite(temp))));

      steps = 100;
      do {
        dest[3] = temp = 12d * _ModelBase._exp(-8d * random.nextDouble());
      } while ((((--steps) > 0) && (temp < 1e-12d))
          || (!(MathUtils.isFinite(temp))));

    } else {
      dest[0] = maxY;
      dest[1] = random.nextDouble() * (minY - maxY);

      steps = 100;
      do {
        dest[2] = temp = 30d * _ModelBase
            ._exp(-7d * (random.nextDouble() * random.nextDouble()));
      } while ((((--steps) > 0) && (temp < 1e-13d))
          || (!(MathUtils.isFinite(temp))));

      steps = 100;
      do {
        dest[3] = temp = -2d * _ModelBase._exp(-random.nextDouble() * 6d);
      } while ((((--steps) > 0) && (temp > -1e-13d))
          || (!(MathUtils.isFinite(temp))));

    }
  }

  /** the parameter guesser */
  private final class __LogisticModelWithOffsetOverLogXParameterGuesser
      extends SamplingBasedParameterGuesser {

    /**
     * Create the parameter guesser
     *
     * @param data
     *          the data
     */
    __LogisticModelWithOffsetOverLogXParameterGuesser(final IMatrix data) {
      super(data, 3, 4);
    }

    /** {@inheritDoc} */
    @Override
    protected final boolean fallback(final double[] points,
        final double[] dest, final Random random) {
      final double[] minMax;
      double temp;

      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY,
          points, random);
      LogisticModelWithOffsetOverLogX._fallback(minMax[0], minMax[1], dest,
          random);

      switch (random.nextInt(3)) {
        case 0: {
          temp = LogisticModelWithOffsetOverLogX._c_x1y1abd(points[0],
              points[1], dest[0], dest[1], dest[3]);
          if (_ModelBase._check(temp, dest[2], 1e-13d)) {
            dest[2] = temp;
          }
          temp = LogisticModelWithOffsetOverLogX._d_x1y1abc(points[0],
              points[1], dest[0], dest[1], dest[2]);
          if (_ModelBase._check(temp, dest[3], 1e-13d)) {
            dest[3] = temp;
          }
          break;
        }
        default: {
          temp = LogisticModelWithOffsetOverLogX._d_x1y1abc(points[0],
              points[1], dest[0], dest[1], dest[2]);
          if (_ModelBase._check(temp, dest[3], 1e-13d)) {
            dest[3] = temp;
          }
          temp = LogisticModelWithOffsetOverLogX._c_x1y1abd(points[0],
              points[1], dest[0], dest[1], dest[3]);
          if (_ModelBase._check(temp, dest[2], 1e-13d)) {
            dest[2] = temp;
          }
        }
      }

      switch (random.nextInt(3)) {
        case 0: {
          temp = LogisticModelWithOffsetOverLogX._a_x1y1bcd(points[0],
              points[1], dest[1], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[0], 1e-13d)) {
            dest[0] = temp;
          }

          temp = LogisticModelWithOffsetOverLogX._b_x1y1acd(points[0],
              points[1], dest[0], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[1], 1e-13d)) {
            dest[1] = temp;
          }
          break;
        }
        default: {
          temp = LogisticModelWithOffsetOverLogX._b_x1y1acd(points[0],
              points[1], dest[0], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[1], 1e-13d)) {
            dest[1] = temp;
          }
          temp = LogisticModelWithOffsetOverLogX._a_x1y1bcd(points[0],
              points[1], dest[1], dest[2], dest[3]);
          if (_ModelBase._check(temp, dest[0], 1e-13d)) {
            dest[0] = temp;
          }

          break;
        }
      }

      return true;
    }

    /** {@inheritDoc} */
    @Override
    protected final void fallback(final double[] dest,
        final Random random) {
      final double[] minMax;
      minMax = _ModelBase._getMinMax(true, this.m_minY, this.m_maxY, null,
          random);
      LogisticModelWithOffsetOverLogX._fallback(minMax[0], minMax[1], dest,
          random);
    }

    /** {@inheritDoc} */
    @Override
    protected final double value(final double x,
        final double[] parameters) {
      return LogisticModelWithOffsetOverLogX.this.value(x, parameters);
    }
  }

}