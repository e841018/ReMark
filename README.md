# ReMark
This repository provides the source code used in *ReMark: Privacy-preserving Fiducial Marker System via Single-pixel Imaging* accepted in [MobiCom 2023](https://sigmobile.org/mobicom/2023/index.html). Please cite our paper if it helps your research.

### Requirements

* Python 3.6 or newer

### Installation

1. Clone this repository.
2. Install Python packages: `<python> -m pip install -r requirements.txt`
3. Install a PyTorch version according to your computing platform.

### Usage

* Identify marker ID from observations:

  ```bash
  cd src
  <python> demo.py
  ```

  * prints details and shows reconstructed images.

* Synthesize datasets:

  ```bash
  cd src
  <python> synthesize.py
  ```

  * generates `dataset/recon_detection/clear_detection_sample=100000.pkl`

    and `dataset/recon_identification/clear_identification_sample=100000.pkl`

* Train reconstruction NNs:

  ```bash
  cd src
  <python> train_recon_detection.py 0
  ```

  * generates `model/recon/model=[dcan_M=72]__train=[clear_detection_sample=100000]__rep=0/model.pt`

  ```bash
  cd src
  <python> train_recon_identification.py 0
  ```

  * generates `model/recon/model=[dcan_M=256]__train=[clear_identification_sample=100000]__rep=0/model.pt`

* Train alignment NNs:

  ```bash
  cd src
  <python> synthesize_and_reconstruct.py
  ```

  * generates `dataset/align/dcan256_snr=15db_sample=100000.pkl`

  ```bash
  cd src
  <python> train_align.py 0
  ```

  * generates `model/align/model=[cnm]__train=[dcan256_snr=15db_sample=100000]__rep=0\model.pt`,
    overwrites if existing.

### Notes on SPI hardware

This repository does not include the code that drives the SPI hardware in the paper. It is highly recommended to replace the outdated DMD and USRP with modern hardware, in which case the old code will not be suitable. Some observations collected by the outdated hardware are available in the `data` directory. The formats are documented in `demo.py`.
