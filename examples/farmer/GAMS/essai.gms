sets 
    old_set1 / element1, element2, element3 /
    old_set2 / element4, element5, element6 /
    all_elements / element1 * element6 /
    new_set(all_elements);

* Create the union of old_set1 and old_set2
new_set(old_set1) = yes;
new_set(old_set2) = yes;

parameters 
    x_1(old_set1) / element1 10, element2 20, element3 30 /
    x_2(old_set2) / element4 40, element5 50, element6 60 /
    x(all_elements);

x(all_elements) = 0;
x(old_set1) = x_1(old_set1);
x(old_set2) = x_2(old_set2);

* Display the sets and parameters to verify their contents
display old_set1, old_set2, new_set;
display x_1, x_2, x;

* Create a simple equation to test the use of x over new_set
variable z;
equation eq;

eq.. z =e= sum(new_set, x(new_set));

model test /all/;

solve test using lp maximizing z;

display z.l;

* Additional test: calculate the sum manually and compare
parameter manual_sum;
manual_sum = sum(new_set, x(new_set));
display manual_sum;

* Check if manual sum equals z
parameter check;
check = abs(manual_sum - z.l) < 1e-6;
display check;