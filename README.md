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

## Installation

### Install on a Conda Environment
Run the following command first to activate the conda environment:
```bash
conda activate <environment_name>
```
If git and pip not already installed in the environment, install both with the following commands:
```bash
conda install git 
conda install pip
```
Now proceed with the General Installation.

### General Installation
Run the following command to install the library:
```bash
pip install git+https://github.com/GGBoss101/climate_lib
```

If installed on a conda environment, the library will be usable in the kernel (in Jupyter Lab, etc) corresponding to the environment.

## Python Usage

### Import The Whole Library

#### Method 1:

##### Import
```python
import climate_lib
```
##### Usage
```python
climate_lib.<file_name>.<function_name>
```
##### Example
```python
import climate_lib
climate_lib.compute.vert_integral(ds, var)
```
#### Method 2:

##### Import
```python
from climate_lib import *
```
##### Usage
```python
<file_name>.<function_name>
```
##### Example
```python
from climate_lib import *
compute.vert_integral(ds, var)
```

### Import Specific Files

#### Method 3:

##### Import
```python
from climate_lib import <file_name>
```
##### Usage
```python
<file_name>.<function_name>
```
##### Example
```python
from climate_lib import <file_name>
compute.vert_integral(ds, var)
```

#### Method 4:

##### Import
```python
from climate_lib.<file_name> import *
```
##### Usage
```python
<function_name>
```
##### Example
```python
from climate_lib.compute import *
vert_integral(ds, var)
```

## Documentation

<a href="https://ggboss101.github.io/climate_lib/">Documentation-Link</a>

## Making Changes

<a href="">WikiPage-Link</a>