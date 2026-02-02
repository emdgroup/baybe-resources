# Tutorials and Resources for BayBE

This repository contains some tutorials and general resources for the usage of [BayBE](https://github.com/emdgroup/baybe), the Bayesian Optimization Framework developed by Merck KGaA, Darmstadt, Germany.

**NOTE**: This repository is updated regularly, and the resources and tutorials here can and will change significantly over time. It might not always reflect the newest versions of BayBE. To verify which version of BayBE is being used, refer to the individual files.

## How to use

This repository contains [marimo](https://marimo.io/) notebooks demonstrating the capabilities of BayBE.
Marimo is an open-source reactive notebook for Python.

To view and use any of the notebooks, execute `marimo edit path/to/notebook.py`. This opens `marimo` in your
browser and enables you to run and execute the notebook.

## Installation

1. Create a python environment using a version of python compatible with BayBE, e.g. via
```
mamba create --yes --name baybe-resources python=3.13
mamba activate baybe-resources
```
if you use `mamba`.
2. Install the additional dependencies from `requirements.txt` via `pip install -r requirements.txt`.
3. Depending on which example you want to investigate, you might need to install the additional dependencies in the corresponding folders.

## 👨🏻‍🔧 Maintainers

- Martin Fitzner (Merck KGaA, Darmstadt, Germany), [Contact](mailto:martin.fitzner@merckgroup.com), [Github](https://github.com/Scienfitz)
- Adrian Šošić (Merck Life Science KGaA, Darmstadt, Germany), [Contact](mailto:adrian.sosic@merckgroup.com), [Github](https://github.com/AdrianSosic)
- Alexander Hopp (Merck KGaA, Darmstadt, Germany) [Contact](mailto:alexander.hopp@merckgroup.com), [Github](https://github.com/AVHopp)


## 📄 License

Copyright 2022-2025 Merck KGaA, Darmstadt, Germany
and/or its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.