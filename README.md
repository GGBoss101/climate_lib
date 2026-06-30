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

## Git Commit Guidelines

We use [Conventional Commits](https://conventionalcommits.org) to automatically manage release notes and Semantic Versioning (SemVer). 

### SemVer-Triggering Commits
These prefixes directly automate version bumps:
* **`fix:`** Triggers a **PATCH** version bump (e.g., `1.0.0` to `1.0.1`). Fixes a bug.
* **`feat:`** Triggers a **MINOR** version bump (e.g., `1.0.0` to `1.1.0`). Introduces a new feature.
* **`!` or `BREAKING CHANGE:`** Triggers a **MAJOR** version bump (e.g., `1.0.0` to `2.0.0`). Introduces breaking changes. 
  * *Example:* `feat!: remove deprecated v1 endpoints`

### Non-Semantic Commits
These prefixes do not trigger a version increment:
* **`chore:`** Maintenance tasks, dependency updates, or tool configurations.
* **`docs:`** Documentation changes only (e.g., README updates).
* **`style:`** Code formatting (spaces, commas, linting) without code meaning changes.
* **`refactor:`** Code changes that neither fix a bug nor add a feature.
* **`perf:`** Code changes that improve performance.
* **`test:`** Adding missing tests or correcting existing tests.
* **`ci:`** Changes to continuous integration files and scripts (e.g., GitHub Actions).

*Example Commit Message:* `feat: add Google login integration`

## Documentation

<a href="https://ggboss101.github.io/climate_lib/">Documentation-Link</a>

## Making Changes

<a href="">WikiPage-Link</a>