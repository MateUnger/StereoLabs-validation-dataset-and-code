# StereoLabs-validation-dataset-and-code
---
## Overview
This is a Python project that implements gait analysis using 3D kinematic data obtained from two different sources (markerless RGB-D, markered) and  also uses statistical analysis to provide a basis for comparing the results.

## Contents
1. **data_structures.py**
    - Definitions of classes
    - Calculations of gait cycle parameters


2. **util.py**
    - utility functions used for analysis and loading data


3. **requirements.txt**
    - packages used to create environment
    - `pip install -r requirements.txt`


4. **main.py**
    - Script tying all others together
    - Loads data, performs gait analysis
    - Produces csv tables contaning the final results of statistical analysis

5. **dataset**
    - Contains the preprocessed and anonymized data used for the study
    - Each subfolder belongs to one participant and contains 3D kinematic data collected using two different walking velocities. For each walking velocity there are two files, one obtained using a IR marker based system (qualisys.npz) and one from an RGB-D sensor (stereo.npz).

## Usage
1. Clone git repository

    ```sh
    git clone https://github.com/MateUngerStereoLabs-validation-dataset-and-code.git
    cd StereoLabs-validation-dataset-and-code
    ```
2. Create & activate virtual environment
    ```sh
    python -m venv venv
    venv\Scripts\activate  # on Linux source venv/bin/activate
    pip install - r requirements.txt
    ```
3. Select the walking velocity being analyzed in `main.py` then run it