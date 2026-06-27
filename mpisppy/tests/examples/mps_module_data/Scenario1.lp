\* Source Pyomo model name=mps_module_test *\

min 
obj:
+3 x(1)
+2 x(2)
+5 y

s.t.

c_l_demand_:
+1 x(1)
+1 x(2)
+1 y
>= 10

bounds
   0 <= x(1) <= 4
   0 <= x(2) <= +inf
   0 <= y <= +inf
end
