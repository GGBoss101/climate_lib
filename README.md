# Climate Library (M. Lague Lab)

`climate_lib` is a Python package for atmospheric and climate-data analysis, providing efficient implementations of common diagnostics used in atmospheric and climate-data analysis tasks.

The library is designed to simplify common climate-data analysis tasks using CESM output and Xarray datasets, allowing researchers to perform routine analyses with concise, reusable code.

The library was developed to simplify frequently used climate-analysis workflows that often require large amounts of repetitive code. Its functions are designed to work naturally with CESM model output while remaining compatible with general atmospheric datasets stored in Xarray format.

## Features

- Vertical integration in pressure and hybrid sigma-pressure coordinates
- Moisture flux divergence calculations
- Linear trend and regression analysis
- Climate-data visualization over North America
- Working directly with Xarray datasets and DataArrays

## Documentation

It can be found at: <a href="https://ggboss101.github.io/climate_lib/">Documentation Link</a>

## Installation

To install on a specific conda environment (optional), run the following commands first to activate it, and install git and pip within the environment:
```bash
conda activate <environment_name>
conda install git pip
```

Run the following command to install the library:

```bash
pip install git+https://github.com/GGBoss101/climate_lib
```

If installed on a conda environment, the library will be usable in the kernel (in Jupyter Lab, etc) corresponding to the environment.

## Python Usage

It can be imported in the following way in python:
```python
from climate_lib import *
```
or for only specific files:
```python
from climate_lib import <file_name>
```