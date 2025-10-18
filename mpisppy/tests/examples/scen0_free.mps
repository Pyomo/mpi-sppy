* As of October 2025, we can't really handle free format. Use lp files instead
NAME          scen0  FREE
ROWS
N  OBJ
L  totalArea
G  requirement__wheat__
G  requirement__corn__
G  requirement__beets__
L  sellBeets
COLUMNS
area__wheat__  totalArea  1
area__wheat__  requirement__wheat__  2
area__wheat__  OBJ  150
area__corn__  totalArea  1
area__corn__  requirement__corn__  2.4
area__corn__  OBJ  230
area__beets__  totalArea  1
area__beets__  requirement__beets__  16
area__beets__  sellBeets  -16
area__beets__  OBJ  260
sell__wheat__  requirement__wheat__  -1
sell__wheat__  OBJ  -170
sell__corn__  requirement__corn__  -1
sell__corn__  OBJ  -150
sell__beets__  requirement__beets__  -1
sell__beets__  sellBeets  1
sell__beets__  OBJ  -36
sell_excess  sellBeets  1
sell_excess  OBJ  -10
buy__wheat__  requirement__wheat__  1
buy__wheat__  OBJ  238
buy__corn__  requirement__corn__  1
buy__corn__  OBJ  210
buy__beets__  requirement__beets__  1
buy__beets__  OBJ  100
RHS
B  totalArea  500
B  requirement__wheat__  200
B  requirement__corn__  240
BOUNDS
UP  BOUND  sell__beets__  6000
ENDATA
