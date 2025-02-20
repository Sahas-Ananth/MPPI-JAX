# Model Predictive Path Integral Control in JAX python

This project implements the MPPI algorithm in Python and accelerates it using GPU programming using `JAX` python. The paper for this algorithm is given [here](https://ieeexplore.ieee.org/document/7989202).

## Installation
The code was run on a:
- **OS:** Ubuntu 22.04
- **GPU:**`NVIDIA GeForce RTX 3070 Laptop GPU` 
- **Driver:** `550.120` 
- **CUDA Version:** `12.4`
- **CUDA Toolkit Version:** `12.8`
```bash
git clone https://github.com/Sahas-Ananth/MPPI-JAX
cd MPPI-JAX
python3 -m venv venv
source venv/bin/activate
# If you just want to use this
pip install -r requirements.txt
pip install .
# else if you want to develop this project
pip install -r requirements-dev.txt
pip install -e .
# end if
```

## Usage

You can run the `src/run_mppi.py` file:
```bash
# To get help
python3 src/run_mppi.py -h
# To set config file
python3 src/run_mppi.py -c <PATH_TO_CONFIG_FILE> # replace everything from < to > with the path
# To use default config found at config/default_MPPI.json is with visualization
# To use default config found at config/default_MPPI_NoVis.json is without visualization (for docker use this)
python3 src/run_mppi.py
```
