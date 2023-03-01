# Checkpoint Utilities

The `convert_hf_checkpoint_to_cerebras.py` file is used to map the HuggingFace GPT-J 6B weights to a Tensorflow checkpoint which can be loaded for either continuous pre-training or fine-tuning with the Cerebras implementation of GPT-J.

## Installation

The following pre-requisites are needed to enable a clean run of the script. We recommend installing an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment and running the following script to get started:

```bash
conda create --name <env> --file requirements.txt
```

## Running the conversion

To convert the checkpoints, activate the created conda environment and run the following command:

```bash
python convert_hf_checkpoint_to_cerebras.py --input_dir </path/to/hf/checkpoint> --ouput_dir </path/to/store/tf/checkpoint>
```

There are two required arguments for this file:

- The `--input_dir` specifies the directory where the HuggingFace checkpoint is stored. If this folder does not exist, it is created during runtime. If the folder exists but does not have a `pytorch_model.bin` file in it, it means that the checkpoint is not available locally, and it will be downloaded from HuggingFace Hub.
- The `--output_dir` specifies the directory where the converted checkpoints will be stored. If this folder does not exist, it is created during runtime.

There are two optional arguments:

- The `--share_embeddings` argument, which is set to True by default. This replicates the original model code by sharing embedding weights with the classifier. You can specify it explicitly to create a checkpoint without shared embeddings.
- The `--debug` argument, which is set to False by default. This helps debug the GPT-J model created from HuggingFace by printing the number of parameters, to verify that you have the right configuration passed in.

__User-Note__: The pre-provided script will download the checkpoint from HuggingFace if it does not exist in the provided `input_dir`. Since the model is very large, it has a checkpoint around 23GB in size. A single checkpoint conversion will take about 110GB RAM and about 10-15mins to run end to end. Please use a system or server with sufficient compute and memory (storage).

## Verifying the conversion

To verify the conversion, activate the created conda environment and run the following command:

```bash
python verify_checkpoint_conversion.py --input_dir </path/to/hf/checkpoint> --ouput_dir </path/to/store/tf/checkpoint>
```

All the arguments passed to this script are the same as the script above. The key difference here is that both the `input_dir`, the `output_dir`, and the corresponding HuggingFace and Cerebras checkpoints should be available during runtime, else the script throws an error and exits.