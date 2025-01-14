$title The Farmer's Problem formulated for GAMS/DECIS (FARM,SEQ=199)

$onText
Linearize the prox term:
  B is xbar
  L is lower bound on x  (which is zero for farmer)
  U is upper bound on x  (which is land=500 for farmer)
  penalty >= B^2 - BL - Bx + Lx
  penalty >= B^2 - BU - Bx + Ux

This model helps a farmer to decide how to allocate
his or her land. The yields are uncertain.


Birge, R, and Louveaux, F V, Introduction to Stochastic Programming.
Springer, 1997.

Keywords: linear programming, stochastic programming, agricultural cultivation,
          farming, cropping
$offText

*$if not set decisalg $set decisalg decism

Set
   crop                                            / wheat, corn, sugarbeets /
   cropr(crop) 'crops required for feeding cattle' / wheat, corn             /
   cropx                                           / wheat
                                                     corn
                                                     beets1 'up to 6000 ton'
                                                     beets2 'in excess of 6000 ton' /;

Parameter
   yield(crop)       'tons per acre'               / wheat         2.5
                                                     corn          3
                                                     sugarbeets   20   /
   plantcost(crop)   'dollars per acre'            / wheat       150
                                                     corn        230
                                                     sugarbeets  260   /
   sellprice(cropx)  'dollars per ton'             / wheat       170
                                                     corn        150
                                                     beets1       36
                                                     beets2       10   /
   purchprice(cropr) 'dollars per ton'             / wheat       238
                                                     corn        210   /
   minreq(cropr)     'minimum requirements in ton' / wheat       200
                                                     corn        240   /
   ph_W(crop)        'ph weight'                   / wheat         0   
                                                     corn          0
                                                     sugarbeets    0   /
   xbar(crop)        'ph average'                  / wheat         0   
                                                     corn          0
                                                     sugarbeets    0   /
   rho(crop)         'ph rho'                      / wheat         0 
                                                     corn          0
                                                     sugarbeets    0   /;

Scalar
   land      'available land'     /  500 /
   maxbeets1 'max allowed'        / 6000 /
   W_on      'activate w term'    /    0 /
   prox_on   'activate prox term' /    0 /;

*--------------------------------------------------------------------------
* First a non-stochastic version
*--------------------------------------------------------------------------
Variable
   x(crop)         'acres of land'
   w(cropx)        'crops sold'
   y(cropr)        'crops purchased'
   yld(crop)       'yield'
   PHpenalty(crop) 'linearized prox penalty'
   negprofit       'objective variable';

Positive Variable x, w, y;

Equation
   profitdef  'objective function'
   landuse    'capacity'
   req(cropr) 'crop requirements for cattle feed'
   ylddef     'calc yields'
   PenLeft(crop) 'left side of linearized PH penalty'
   PenRight(crop) 'right side of linearized PH penalty'
   beets      'total beet production';

$onText
The YLD variable and YLDDEF equation isolate the stochastic
YIELD parameter into one equation, making the DECIS setup
somewhat easier than if we would substitute YLD out of
the model.
$offText

profitdef..    negprofit =e= + sum(crop,  plantcost(crop)*x(crop))
                       +  sum(cropr, purchprice(cropr)*y(cropr))
                       -  sum(cropx, sellprice(cropx)*w(cropx))
                       +  W_on * sum(crop, ph_W(crop)*x(crop))
                       +  prox_on * sum(crop, 0.5 * rho(crop) * PHpenalty(crop));

landuse..      sum(crop, x(crop)) =l= land;

ylddef(crop).. yld(crop) =e= yield(crop)*x(crop);

req(cropr)..   yld(cropr) + y(cropr) - sum(sameas(cropx,cropr),w(cropx)) =g= minreq(cropr);

beets..        w('beets1') + w('beets2') =l= yld('sugarbeets');

PenLeft(crop).. sqr(xbar(crop)) + xbar(crop)*0 + xbar(crop) * x(crop) + land * x(crop) =g= PHpenalty(crop);
PenRight(crop).. sqr(xbar(crop)) - xbar(crop)*land - xbar(crop)*x(crop) + land * x(crop) =g= PHpenalty(crop);

PHpenalty.lo(crop) = 0;
PHpenalty.up(crop) = max(sqr(xbar(crop) - 0), sqr(land - xbar(crop)));

w.up('beets1') = maxbeets1;

x.lo(crop) = 0;
x.up(crop) = land;

Model simple / profitdef, landuse, req, beets, ylddef, PenLeft, PenRight /;

Option LP = Cplex;
solve simple using lp minimizing negprofit;
