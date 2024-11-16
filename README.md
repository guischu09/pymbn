# README #

### Repository Description ###

This repository contains an under development full python implementation of the Multiple Sampling (MS) scheme for contructing stable Metabolic Brain Networks proposed at: https://doi.org/10.1101/2021.03.16.435674

### Setup ###
**Python setup (using Miniconda/Anaconda):**

1. If you do not have Miniconda/Anaconda installed, go to [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and download it.

2. Clone/download this repository.

3. Open a terminal and navigate to the cloned/downloaded folder. Afterwards, type the following command:

> conda create -n pymbn python=3.10 pip

4. Now, install the dependencies with the following commands:

> conda activate pymbn
> pip install -r requirements.txt --no-deps

5. You may need to install the follwing packages:
> sudo apt-get install python3-pyqt5.qtsvg libxcb-xinerama0

### Running ###

1. Open a terminal and navigate to the cloned/downloaded folder. Afterwards, type the following commands:
> conda activate pymbn
2. Generate results running the main scrpit:
> python main.py

Check the obtained results at the results and outputs created directories.

### Methods ###

Check our paper for the details: Stable brain PET metabolic networks using a multiple sampling scheme - https://doi.org/10.1101/2021.03.16.435674

### Contact ###

guischu09@gmail.com - Guilherme Schu

