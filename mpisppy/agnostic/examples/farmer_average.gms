$title The Farmer s Problem formulated for GAMS/DECIS (FARM,SEQ=199)

$onText
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
                                                     corn        240   /;

Scalar
   land      'available land' /  500 /
   maxbeets1 'max allowed'    / 6000 /;

*--------------------------------------------------------------------------
* First a non-stochastic version
*--------------------------------------------------------------------------
Variable
   x(crop)    'acres of land'
   w(cropx)   'crops sold'
   y(cropr)   'crops purchased'
   yld(crop)  'yield'
   profit     'objective variable';

Positive Variable x, w, y;

Equation
   profitdef  'objective function'
   landuse    'capacity'
   req(cropr) 'crop requirements for cattle feed'
   ylddef     'calc yields'
   beets      'total beet production';

$onText
The YLD variable and YLDDEF equation isolate the stochastic
YIELD parameter into one equation, making the DECIS setup
somewhat easier than if we would substitute YLD out of
the model.
$offText

profitdef..    profit =e= - sum(crop,  plantcost(crop)*x(crop))
                       -    sum(cropr, purchprice(cropr)*y(cropr))
                       +    sum(cropx, sellprice(cropx)*w(cropx));

landuse..      sum(crop, x(crop)) =l= land;

ylddef(crop).. yld(crop) =e= yield(crop)*x(crop);

req(cropr)..   yld(cropr) + y(cropr) - sum(sameas(cropx,cropr),w(cropx)) =g= minreq(cropr);

beets..        w('beets1') + w('beets2') =l= yld('sugarbeets');

x.up(crop) = land;
w.up('beets1') = maxbeets1;
$onText
__InsertPH__here_Model_defined_three_lines_later
$offText

Model simple / profitdef, landuse, req, beets, ylddef /;

solve simple using lp maximizing profit;