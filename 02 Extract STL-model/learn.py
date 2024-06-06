#!\Python27\python
import sys

import pytest
import synth
import pandas
import numpy
import sys, getopt
import os
from scorer import set_beta_gamma, get_beta, get_gamma

# To run:
# py -2.7 learn.py -i 3 -o "gradient"

result_folder = "Result" # If folder exists, a number is added to the end
LOGFILE = "Result_"

# Tune TeLEx parameters
BETA = 80
GAMMA = 0.01

# Choose STL-templates to run
run_for_variables = [
    'giver_RHand',
    'd_giver_RHand',
    'dd_giver_RHand',
    'taker_RHand',
    'd_taker_RHand',
    'dd_taker_RHand',
    'relative',
    'd_relative',
    'dd_relative',
]

run_for_temporal_opperators = [
    'G',
    'F',
    ]

templogicdata = []

if 'F' in run_for_temporal_opperators and 'giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        ]
if 'F' in run_for_temporal_opperators and 'd_giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        ]
if 'F' in run_for_temporal_opperators and 'dd_giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
    ]
if 'F' in run_for_temporal_opperators and 'taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        ]
if 'F' in run_for_temporal_opperators and 'd_taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
    ]
if 'F' in run_for_temporal_opperators and 'dd_taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
    ]
if 'F' in run_for_temporal_opperators and 'relative' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        ]
if 'F' in run_for_temporal_opperators and 'd_relative' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        ]
if 'F' in run_for_temporal_opperators and 'dd_relative' in run_for_variables:
    templogicdata = templogicdata + [
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'F[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
        'F[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
        'F[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (x_giver_RHand > lb? -1;1 & x_giver_RHand < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (y_giver_RHand > lb? -1;1 & y_giver_RHand < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (z_giver_RHand > lb? -1;1 & z_giver_RHand < ub? -1;1)',
        ]
if 'G' in run_for_temporal_opperators and 'd_giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dx_giver_RHand > lb? -5;5 & dx_giver_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dy_giver_RHand > lb? -5;5 & dy_giver_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dz_giver_RHand > lb? -5;5 & dz_giver_RHand < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'dd_giver_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddx_giver_RHand > lb? -5;5 & ddx_giver_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddy_giver_RHand > lb? -5;5 & ddy_giver_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddz_giver_RHand > lb? -5;5 & ddz_giver_RHand < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (x_taker_RHand > lb? -1;1 & x_taker_RHand < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (y_taker_RHand > lb? -1;1 & y_taker_RHand < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (z_taker_RHand > lb? -1;1 & z_taker_RHand < ub? -1;1)',
        ]
if 'G' in run_for_temporal_opperators and 'd_taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dx_taker_RHand > lb? -5;5 & dx_taker_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dy_taker_RHand > lb? -5;5 & dy_taker_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (dz_taker_RHand > lb? -5;5 & dz_taker_RHand < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'dd_taker_RHand' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddx_taker_RHand > lb? -5;5 & ddx_taker_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddy_taker_RHand > lb? -5;5 & ddy_taker_RHand < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] (ddz_taker_RHand > lb? -5;5 & ddz_taker_RHand < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'relative' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {x_giver_RHand - x_taker_RHand} > lb? -1;1 & {x_giver_RHand - x_taker_RHand} < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {y_giver_RHand - y_taker_RHand} > lb? -1;1 & {y_giver_RHand - y_taker_RHand} < ub? -1;1)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {z_giver_RHand - z_taker_RHand} > lb? -1;1 & {z_giver_RHand - z_taker_RHand} < ub? -1;1)',
        ]
if 'G' in run_for_temporal_opperators and 'd_relative' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {dx_giver_RHand - dx_taker_RHand} > lb? -5;5 & {dx_giver_RHand - dx_taker_RHand} < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {dy_giver_RHand - dy_taker_RHand} > lb? -5;5 & {dy_giver_RHand - dy_taker_RHand} < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {dz_giver_RHand - dz_taker_RHand} > lb? -5;5 & {dz_giver_RHand - dz_taker_RHand} < ub? -5;5)',
        ]
if 'G' in run_for_temporal_opperators and 'dd_relative' in run_for_variables:
    templogicdata = templogicdata + [
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddx_giver_RHand - ddx_taker_RHand} > lb? -5;5 & {ddx_giver_RHand - ddx_taker_RHand} < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddy_giver_RHand - ddy_taker_RHand} > lb? -5;5 & {ddy_giver_RHand - ddy_taker_RHand} < ub? -5;5)',
        'G[t1? 0.0;1.0, t2? 0.0;1.0] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
        'G[t1? 0.5;1.5, t2? 0.5;1.5] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
        'G[t1? 1.0;2.0, t2? 1.0;2.0] ( {ddz_giver_RHand - ddz_taker_RHand} > lb? -5;5 & {ddz_giver_RHand - ddz_taker_RHand} < ub? -5;5)',
    ]

learning_data = "learning_data"
validation_data = "validation_data"

#expected_robustness = 0.055 # Calculate the peak functions peak with beta
#expected_robustness = 5 # Because right now I don't care

@pytest.mark.parametrize("tlStr", templogicdata)
def test_stl(tlStr, optmethod = "gradient"):
    

    print(tlStr)
    (stlsyn, value, dur, ppvalues) = synth.synthSTLParam2(tlStr, learning_data, optmethod)
    print(" Synthesized STL formula: {}\n Theta Optimal Value: {}\n Optimization time: {}\n".format(stlsyn, value, dur))

    print("Testing result of synthesized STL on learning traces")
    (bres, qres) = synth.verifySTL(stlsyn, learning_data)
    #print(" Test result of synthesized STL on each learning trace: {}\n Robustness Metric Value: {}".format(bres, qres))
    passed_test_ratio_learning = float(sum(bres))/len(bres)
    avg_robustness_learning = sum(qres)/len(qres)
    print(" Percentage of 'True' synthesized STL on learning data: {}\n Average Robustness Metric Value: {}\n".format( passed_test_ratio_learning*100, avg_robustness_learning ))

    print("Testing result of synthesized STL on validation traces")
    (bres, qres) = synth.verifySTL(stlsyn, validation_data)
    #print(" Test result of synthesized STL on each validation trace: {}\n Robustness Metric Value: {}".format(bres, qres))
    passed_test_ratio_validation = float(sum(bres))/len(bres)
    avg_robustness_validation = sum(qres)/len(qres)
    print(" Percentage of 'True' synthesized STL on validation data: {}\n Average Robustness Metric Value: {}\n".format( passed_test_ratio_validation*100, avg_robustness_validation ))

    return stlsyn,value,dur,passed_test_ratio_learning,avg_robustness_learning,passed_test_ratio_validation,avg_robustness_validation, ppvalues
 

def main(argv):
    global result_folder
    # Create a new folder for results
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # Append a number to the folder name to get unique result folder
    else:
        folder_number = 1
        while os.path.exists(result_folder + "_{}".format(int(folder_number))):
            folder_number += 1
        result_folder = result_folder + "_{}".format(int(folder_number))
        os.makedirs(result_folder)


    # Args if none given:
    itercount = 2 
    optmethod = "gradient"

    
    
    try: 
        opts,args = getopt.getopt(argv, "hi:o:",["itercount=","optmethod="])
    except getopt.GetoptError:
        print( 'python test_scale.py -i <number of times to iterate each synthesis task to compute mean runtime> -o <opt-method>' )
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print( 'python [this filename].py -i <number of times to iterate each synthesis task to compute mean runtime> -o <opt-method>')
            print( 'Valid opt-methods: \"gradient\", \"nogradient\"')
            sys.exit()
        elif opt in ("-i", "--itercount"):
            itercount = int(arg)
        elif opt in ("-o", "--optmethod"):
            optmethod = arg

    
    logfile = result_folder + "/" + LOGFILE + optmethod + "_" + str(itercount) + ".log"
    #stlonlyfile = result_folder + STL_LOGFILE + optmethod + "_" + str(itercount) + ".log"
    STL_eq100_file = result_folder + "/STL_100.log"
    STL_95_100_file= result_folder + "/STL_95-100.log"
    STL_90_95_file = result_folder + "/STL_90-95.log"
    STL_85_90_file = result_folder + "/STL_85-90.log"
    STL_80_85_file = result_folder + "/STL_80-85.log"
    STL_75_80_file = result_folder + "/STL_75-80.log"
    STL_70_75_file = result_folder + "/STL_70-75.log"
    STL_lt70_file  = result_folder + "/STL_lt70.log"

    STL_eq100_file_b = result_folder + "/STL_100_b.log"
    STL_95_100_file_b = result_folder + "/STL_95-100_b.log"
    STL_90_95_file_b = result_folder + "/STL_90-95_b.log"
    STL_85_90_file_b = result_folder + "/STL_85-90_b.log"
    STL_80_85_file_b = result_folder + "/STL_80-85_b.log"
    STL_75_80_file_b = result_folder + "/STL_75-80_b.log"
    STL_70_75_file_b = result_folder + "/STL_70-75_b.log"
    STL_lt70_file_b = result_folder + "/STL_lt70_b.log"

    about_file = result_folder + "/about.log"

    print("Writing to logfiles in directory: {}".format(result_folder))

    f0=open(about_file, 'a')
    f0.write("Full results of optimization found in '" +logfile+ "'\n")
    f0.write("\nResulting STL formula are separated by % data fit on validation data. These are formated as a nested list, to copy-paste into the controller. \n")
    f0.write("\nResults contain parameters:\n " + str(run_for_variables) + "\nWith temporal opperators:\n " + str(run_for_temporal_opperators))
    f0.write("\nBeta = " + str(BETA) + "\nGamma = " + str(GAMMA))
    f0.write("\nItterations for each template: " + str(itercount))
    f0.close()

    f1=open(logfile, 'a')
    f1.write("========================*START OF RUN*===============================\n")
    f1.write(" Data sets:\n Learning data: {}\n Validation data: {}\n".format(learning_data, validation_data))
    f1.write("======================================================================\n")
    #template = "{0:5}|{1:100}|{2:41}|{3:41}\n"
    #f1.write(template.format(" ", "   Result ", "  Learning data  ", "  Validation data  ") )
    template = "{0:5}|{1:5}|{2:5}|{4:20}|{5:20}|{6:20}|{7:20}|{3:110}\n" 
    f1.write(template.format("ID", "Beta", "Gamma", "   Result ", " Avg Robustness ", " Data fit (%) ", " Avg Robustness ", " Data fit (%) ") )
    f1.close()

    runtime = {}
    rhovalues = {}
    doneflag = {}

    learning_avg_robustness_values = {}
    learning_data_fit_values = {}
    validtion_avg_robustness_values = {}
    validtion_data_fit_values = {}

    result_STL = {}

    for templ in templogicdata:
        runtime[templ] = []
        rhovalues[templ] = []
        doneflag[templ] = False
        
        learning_avg_robustness_values[templ] = []
        learning_data_fit_values[templ] = []
        validtion_avg_robustness_values[templ] = []
        validtion_data_fit_values[templ] = []
        
        result_STL[templ] = []


    for templ in templogicdata:

        f1=open(logfile, 'a')
        #f1.write("========================Optimization-of===============================\n")
        #f1.write(" Template: {}\n".format(templ))
        #f1.write("======================================================================\n")
        #template = "{0:5}|{1:100}|{2:41}|{3:41}\n"
        #f1.write(template.format(" ", "   Result ", "  Learning data  ", "  Validation data  ") )
        template = "{0:5}|{1:5}|{2:5}|{4:20}|{5:20}|{6:20}|{7:20}|{3:100}\n" 
        f1.write(template.format("ID", "Beta", "Gamma", "   Result ", " LAvg Robustness ", " LData fit (%) ", " VAvg Robustness ", " VData fit (%) ") )
        f1.close()

        i = 0
        attempts = 0
        while (i <  itercount) and (attempts < itercount):
            
            set_beta_gamma(BETA, GAMMA)
            tries = 0
            got_result = False
            while tries <= 2 and not(got_result):
                try:
                    stlsyn, value, dur, learn_data_fit, learning_avg_robustness, validation_data_fit, validation_avg_robustness, ppvalues = test_stl(templ, optmethod)
                    print("Finished {} (Iter: {})\n Synthesized STL formula: {}\n Cost Value: {}\n Optimization time: {}\nOn learn data:\n Average robustness: {}\n data fit: {}%\nOn validation data:\n Average robustness: {}\n data fit: {}%\n".format(templ, i, stlsyn, value, dur ,learning_avg_robustness, learn_data_fit*100, validation_avg_robustness, validation_data_fit*100))
                    got_result = True
                except ValueError as er:
                    print(er)
                    print("aborting")
                    tries += 1
                    got_result = False
                except KeyboardInterrupt:
                    # quit
                    sys.exit()
                except: # any other error
                    print("unexpected error, probably overflow, aborting")
                    tries += 1
                    got_result = False

            if got_result == False:
                continue

                #learn_data_fit = 0.0 # Force discard data
                #learning_avg_robustness = 1000000.0
                ##if attempts > itercount + 3:
                #try:
                #    print("trying non-gradient optimization")
                #    stlsyn, value, dur, learn_data_fit, learning_avg_robustness, validation_data_fit, validation_avg_robustness = test_stl(templ, "nogradient")
                #except:
                #    print("Giving up on this one")

            # print("({} < {}) *5 and {} < 0.75".format(learning_avg_robustness, expected_robustness, learn_data_fit))
            #if (learning_avg_robustness < expected_robustness * 5) and (learn_data_fit > 0.75):
            print(" Result kept")

            runtime[templ].append(dur)
            rhovalues[templ].append(value)
                
            learning_avg_robustness_values[templ].append(learning_avg_robustness)
            learning_data_fit_values[templ].append(learn_data_fit)
            validtion_avg_robustness_values[templ].append(validation_avg_robustness)
            validtion_data_fit_values[templ].append(validation_data_fit)

            result_STL[templ].append(templ)

            f1=open(logfile, 'a')
            f1.write(template.format(i, get_beta(), get_gamma(), stlsyn, learning_avg_robustness, learn_data_fit *100, validation_avg_robustness, validation_data_fit *100))
            f1.close()

            # Write STL result to correct file
            if (validation_data_fit == 1): 
                f2=open(STL_eq100_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_eq100_file_b, 'a')
                
            elif (validation_data_fit < 1) and (validation_data_fit >= 0.95):
                f2=open(STL_95_100_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_95_100_file_b, 'a')
            elif (validation_data_fit < 0.95) and (validation_data_fit >= 0.90):
                f2=open(STL_90_95_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_90_95_file_b, 'a')
            elif (validation_data_fit < 0.90) and (validation_data_fit >= 0.85):
                f2=open(STL_85_90_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_85_90_file_b, 'a')
            elif (validation_data_fit < 0.85) and (validation_data_fit >= 0.80):
                f2=open(STL_80_85_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_80_85_file_b, 'a')
            elif (validation_data_fit < 0.80) and (validation_data_fit >= 0.75):
                f2=open(STL_75_80_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_75_80_file_b, 'a')
            elif (validation_data_fit < 0.75) and (validation_data_fit >= 0.70):
                f2=open(STL_70_75_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_70_75_file_b, 'a')
            elif (validation_data_fit < 0.70):
                f2=open(STL_lt70_file, 'a')
                f2.write('"{}", ...\n'.format(stlsyn))
                f2.close()
                f3=open(STL_lt70_file_b, 'a')
            
            if templ[0] == "F":
                f3.write('["F", ')
            elif templ[0] == "G":
                f3.write('["G", ')
            f3.write("{}, {}, {}, ".format(ppvalues['t1'], ppvalues['t2'], ppvalues['lb']) )

            if ('(x_giver_RHand > lb' in templ):
                f3.write("'giver_x', ")
            if ('(dx_giver_RHand > lb' in templ):
                f3.write("'giver_dx', ")
            if ('(ddx_giver_RHand > lb' in templ):
                f3.write("'giver_ddx', ")
                    
            if ('(y_giver_RHand > lb' in templ):
                f3.write("'giver_y', ")
            if ('(dy_giver_RHand > lb' in templ):
                f3.write("'giver_dy', ")
            if ('(ddy_giver_RHand > lb' in templ):
                f3.write("'giver_ddy', ")
                    
            if ('(z_giver_RHand > lb' in templ):
                f3.write("'giver_z', ")
            if ('(dz_giver_RHand > lb' in templ):
                f3.write("'giver_dz', ")
            if ('(ddz_giver_RHand > lb' in templ):
                f3.write("'giver_ddz', ")
                
            if ('(x_taker_RHand > lb' in templ):
                f3.write("'taker_x', ")
            if ('(dx_taker_RHand > lb' in templ):
                f3.write("'taker_dx', ")
            if ('(ddx_taker_RHand > lb' in templ):
                f3.write("'taker_ddx', ")
                    
            if ('(y_taker_RHand > lb' in templ):
                f3.write("'taker_y', ")
            if ('(dy_taker_RHand > lb' in templ):
                f3.write("'taker_dy', ")
            if ('(ddy_taker_RHand > lb' in templ):
                f3.write("'taker_ddy', ")
                    
            if ('(z_taker_RHand > lb' in templ):
                f3.write("'taker_z', ")
            if ('(dz_taker_RHand > lb' in templ):
                f3.write("'taker_dz', ")
            if ('(ddz_taker_RHand > lb' in templ):
                f3.write("'taker_ddz', ")

            if ('{x_giver_RHand - x_taker_RHand}' in templ):
                f3.write("'relative_x', ")
            if ('{dx_giver_RHand - dx_taker_RHand}' in templ):
                f3.write("'relative_dx', ")
            if ('{ddx_giver_RHand - ddx_taker_RHand}' in templ):
                f3.write("'relative_ddx', ")

            if ('{y_giver_RHand - y_taker_RHand}' in templ):
                f3.write("'relative_y', ")
            if ('{dy_giver_RHand - dy_taker_RHand}' in templ):
                f3.write("'relative_dy', ")
            if ('{ddy_giver_RHand - ddy_taker_RHand}' in templ):
                f3.write("'relative_ddy', ")
                    
            if ('{z_giver_RHand - z_taker_RHand}' in templ):
                f3.write("'relative_z', ")
            if ('{dz_giver_RHand - dz_taker_RHand}' in templ):
                f3.write("'relative_dz', ")
            if ('{ddz_giver_RHand - ddz_taker_RHand}' in templ):
                f3.write("'relative_ddz', ")

            f3.write("{}],".format(ppvalues['ub']))
            f3.write("\n")
            f3.close()

            attempts = attempts + 1

        doneflag[templ] = True

        


    #f1=open(logfile, 'a')
    #f1.write("========================FINAL=========================================\n")
    #f1.write("               Optmethod {}\n".format(optmethod))
    #f1.write("======================================================================\n")
    #f1.write("                Averaging over {} Iterations\n".format(itercount))
    #f1.write("======================================================================\n")
    #template = "{0:5}|{1:10}|{2:25}|{3:25}|{4:20}|{5:20}\n" 
    #f1.write(template.format("ID", " #Params", "     Mean Runtime", "     Variance in Runtime", "  Rho Mean ", " Rho Var ") )
    #i = 1
    #for templ in templogicdata:
    #    f1.write("----------------------------------------------------------------------\n")
    #    f1.write(template.format(i, 2*i, numpy.mean(runtime[templ]), numpy.var(runtime[templ]), numpy.mean(rhovalues[templ]), numpy.var(rhovalues[templ]) ))
    #    i = i+1
    #f1.write("======================================================================\n")
    #f1.close()

    #print("Mean Robustness Value: {}, Variance: {}".format(numpy.mean(rhovalues), numpy.var(rhovalues) ) )


if __name__ == "__main__":
    main(sys.argv[1:])

