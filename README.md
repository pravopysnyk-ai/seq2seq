# Sequence to Sequence Model Training & Evaluation

This repository provides code for training and evaluating models for grammatical error correction for the Ukrainian language.

It is mainly based on `PyTorch` and `transformers`.

## Requirements

Our language models were trained in a CUDA-enabled environment. To set everything up, follow the [instructions](https://docs.nvidia.com/cuda/) from the official NVIDIA website.

## Installation

First, you need to initialize the helper submodule directory:

`git submodule update --init`

Then, run the following command to install all necessary packages:

`pip install -r requirements.txt`

The project was tested using Python 3.7.

## Training

To train the models, first initialize the trainer with specified paths to the input data and the model name.

`trainer = PravopysnykTrainer(your_source_file, your_target_file, your_model_name)`

Then, run the driver function to launch the process. You need to have a valid [HuggingFace](https://huggingface.co/) write access token to save your model.

`trainer.main()`
