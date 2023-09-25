#!/bin/bash

git clone https://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@github.com/ChinmayMittal/COL761.git ${pwd}
cd ./COL761/A2/
module purge
module load compiler/gcc/9.1.0
module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/pandas/0.23.4/gnu