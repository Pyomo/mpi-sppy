$title Extensive form of a Transportation Problem (TRNSPORT,SEQ=1)

$onText
This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

Keywords: linear programming, transportation problem, scheduling
$offText

Set
   i 'canning plants' / seattle,  san-diego /
   j 'markets'        / new-york, chicago, topeka /
   scens 'scenarios'  / bad,      average, good/;

Parameter
   a(i) 'capacity of plant i in cases'
        / seattle    350
          san-diego  600 /

   b_average(j) 'average demand at market j in cases'
        / new-york   325
          chicago    300
          topeka     275 /
          
   stoch_prop(scens) 'proportion of demand for a stochastic scenario'
        / bad        0.8
          average    1.0
          good       1.2 /
          
   b_stoch(j, scens) 'demand at market j for the scenario scens';

b_stoch(j,scens) = b_average(j)*stoch_prop(scens);

Table d(i,j) 'distance in thousands of miles'
              new-york  chicago  topeka
   seattle         2.5      1.7     1.8
   san-diego       2.5      1.8     1.4;

Scalar f 'freight in dollars per case per thousand miles' / 90 /;

Parameter 
   c(i,j) 'transport cost in thousands of dollars per case'
   cost_y(j) 'costpenalty of the slack penalty';
c(i,j) = f*d(i,j)/1000;
cost_y(j) = 20;

Variable
   x(i,j) 'shipment quantities in cases'
   z_stoch(scens) 'total transportation and slack costs in thousands of dollars for a scenario'
   z_average      'total average transportation and slack costs in thousands of dollars'
   y(j, scens)    'slack penalty for the demand not supplied';

Positive Variable x;
Positive Variable y;
x.up(i,j) = 1000;

Equation
   cost_stoch(scens)      'define objective function for a scenario'
   cost_average    'define average objective function'
   supply(i) 'observe supply limit at plant i'
   demand(j, scens) 'satisfy demand at market j';

cost_stoch(scens)..      z_stoch(scens) =e= sum((i,j), c(i,j)*x(i,j)) + sum(j, cost_y(j)*y(j,scens));

cost_average..      z_average =e= sum(scens, z_stoch(scens))/3;

supply(i).. sum(j, x(i,j)) =l= a(i);

demand(j, scens).. sum(i, x(i,j)) + y(j, scens) =e= b_stoch(j, scens);

$onText
__InsertPH__here_Model_defined_three_lines_later
$offText

Model transport / all /;

solve transport using lp minimizing z_average;
