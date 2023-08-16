# ReMark
This repository provides the source code used in *ReMark: Privacy-preserving Fiducial Marker System via Single-pixel Imaging* accepted in [MobiCom 2023](https://sigmobile.org/mobicom/2023/index.html). Please cite our paper if it helps your research.

### Requirements

* Python 3.6 or newer

### Installation

1. Clone this repository.
2. Install Python packages: `<python> -m pip install -r requirements.txt`
3. Install a PyTorch version according to your computing platform.

### Usage

* Identify marker ID from observations: `demo.py`
* Synthesize datasets: TODO
* Train reconstruction NNs: TODO
* Train alignment NNs: TODO

### Notes on SPI hardware

This repository does not include the code that drives the SPI hardware in the paper. It is highly recommended to replace the outdated DMD and USRP with modern hardware, in which case the old code will not be suitable. Some observations collected by the outdated hardware are available in the `hardware` directory. The formats are documented in `demo.py`.
