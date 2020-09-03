#!/bin/bash
# dlw aug 2020; create some scenarios for JP

    python -m mape_maker -xf "mape_maker/samples/wind_total_forecast_actual_070113_063015.csv" -f "actuals" -n 200 -bp "ARMA" -o "wind_actuals_ARMA" -s 1234 -ss "2014-7-12 00:00:00" -se "2014-7-13 00:00:00"
