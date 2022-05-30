# Decoding EEG with Spiking Neural Networks on Neuromorphic Hardware

This package is the PyTorch implementation of the Spiking Neural Network for decoding EEG on Neuromorphic Hardware.

## Software Installation ##
* Python 3.7 or higher
* PyTorch 1.2 (with CUDA 10.0)
* NxSDK 0.9

A CUDA enabled GPU is not required but preferred for training. 
The results in the paper are generated from models trained using both Nvidia Tesla K40c and Nvidia GeForce RTX 2080Ti.

Intel's neuromorphic library NxSDK is only required for SNN deployment on the Loihi neuromorphic chip. 
If you are interested in deploying the trained SNN on Loihi, please contact the [Intel Neuromorphic Lab](https://www.intel.com/content/www/us/en/research/neuromorphic-community.html).

## Dataset ##
We provide here the implementation for training the SNN on eegmmidb dataset. The dataset can be downloaded from [this link](https://physionet.org/content/eegmmidb/1.0.0/). Please download the files into a folder named 'data' in your working directory. 

## Example Usage ##

#### 1. Preprocessing the dataset ####

To preprocess and save the preprocessed data, run the following

```bash
cd <Dir>/<Project Name>/utils
python utility.py
```

This will preprocess the dataset and save it into a folder named "eegmmidb_slice_norm"

#### 2. Train the SNN ####

To train the SNN on the eegmmidb dataset, execute the following commands:

```bash
cd <Dir>/<Project Name>/eegmmidb
python train.py
```

This will automatically train the SNN and display the progress of training. 

#### 3. Deploy the trained SNN on Loihi ####

To evaluate SNN realization on Loihi, first run the following to train the simplified model for Loihi:

```bash
cd <Dir>/<Project Name>/eegmmidb_loihi/offline_train
python train.py
```

Then execute the following commands to start testing on Loihi:

```bash
cd <Dir>/<Project Name>/eegmmidb_loihi/online_loihi_inf
KAPOHOBAY=1 python online_loihi_inf.py
```

This will test the model that is trained on the GPU and deployed on Loihi. To run the code correctly, `MODEL_DIR` value in the script needs to be changed to the directory that stores the trained model.
