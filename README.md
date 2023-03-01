# Cerebras Model Zoo

## Introduction

This repository contains examples of common deep learning models that can be trained on Cerebras hardware. These models demonstrate the best practices for coding a model targeted at the Cerebras hardware so that you can take full advantage of this new powerful compute engine.

In order to get started with running your models on a CS system, please refer to the [Developer Documentation](https://docs.cerebras.net/en/latest/index.html) along with this readme.

## Supported frameworks

We support the models developed in [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/). To get more info on framework specific workflow, please refer to the developer docs listed below:

- [PyTorch CS workflow](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-cs-workflow.html)
- [TensorFlow CS workflow](https://docs.cerebras.net/en/latest/tensorflow-docs/cs-tf-workflow.html)

## Basic workflow

When you are targeting the Cerebras CS system for your neural network jobs, please follow the quick start links from the developer docs listed below to compile, validate and train the models in this ModelZoo for the framework of your choice.

- [PyTorch Quickstart](https://docs.cerebras.net/en/latest/getting-started/cs-pytorch-qs.html)
- [TensorFlow Quickstart](https://docs.cerebras.net/en/latest/getting-started/cs-tf-quickstart.html)
- [Weight Streaming Quickstart](https://docs.cerebras.net/en/latest/getting-started/weight-streaming-quickstart.html)

For advanced use cases and porting your existing code from TF or PyTorch to be Cerebras compatible, the high-level workflow can be described as:

1. Port your code to CS in one of the [supported frameworks](#supported-frameworks).
   - For PyTorch use `cerebras.framework.torch`.
   - For TensorFlow use `CerebrasEstimator`.
2. Prepare input data ensuring that you pre-process the input data by sharding, shuffling, prefetching, interleaving, repeating, batching, etc., in a proper order.
3. Compile your code on CPU to optimize your code for your specific CS system early on.<sup>[*](#footnotes)</sup>
4. Run your compiled code on the CS system.

## Execution modes

On the Cerebras Wafer Scale Engine (WSE) you can run neural networks of different model sizes. Cerebras Software supports different execution modes to efficiently run such variety of models.

The execution mode refers to how the Cerebras runtime loads your neural network model onto the Cerebras Wafer Scale Engine (WSE). Two execution modes are supported:

- **Layer pipelined**: In this mode all the layers of the network are loaded altogether onto the Cerebras WSE. This mode is selected for neural network models of small to medium sized models (with less than a billion parameters).

- **Weight streaming**: In this mode one layer of the neural network model is loaded at a time. This layer-by-layer mode is used to run extremely large models (with billions to trillions of parameters).

You can get more information about this on the developer page section on [Cerebras Execution Modes](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html#cerebras-execution-modes)

## Optimizations for Cerebras hardware

We provide various features to speed up the training by leveraging properties of the Cerebras hardware. Following are the key features we provide:

- [Variable Tensor Shape in PyTorch](#pytorch-variable-tensor-shape-vts)
- [Variable Sequence Length in TensorFlow](#tensorflow-variable-sequence-length-vsl)
- [Multi-replica](#multi-replica-mode)

For general optimization techniques, please refer to the [Performance Best Practices page](https://docs.cerebras.net/en/latest/general/performance-optimization.html).

### PyTorch Variable Tensor Shape (VTS)

Variable Tensor Shape (VTS) is a feature that allows computations on the CS system running in pipeline mode to process tensors which vary in shape from one element of a batch to the next. This helps in accommodating input data with heterogeneous sequence length, allowing users to strip away large padding samples on smaller sequences. This leads to less wasted computations and improves the training time of the models.

To learn more about VTS, visit the developer doc page on the same topic [here](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-vts.html).

### TensorFlow Variable Sequence Length (VSL)

Conceptually same as VTS, VSL is a legacy name that we support on the TensorFlow models. VSL is limited in its generality and is currently in the process of being replaced by VTS.

To learn more about VSL, visit the developer doc page on the same topic [here](https://docs.cerebras.net/en/latest/tensorflow-docs/tf-vsl.html).

### Multi-replica mode

Multi-replica Data Parallel Training is a feature that the Cerebras compiler uses to create several copies (replicas) of the same model to run data parallel training. This is similar to how multiple GPUs are used to accelerate training of a single model.

In the background, the compiler ensures that these replicas are initialized with the same weights, and during the training, the weights across all replicas are synchronized after every batch.

A single trained model is available at the conclusion of multi-replica data parallel training. This multi-replica data parallel feature can be used only for training the model.

To learn more about multi-replica, please visit the developer doc page on the same topic [here](https://docs.cerebras.net/en/latest/general/multi-replica-data-parallel-training.html).

## Models in this repository

| Model | Layer Pipeline mode | Weight Streaming mode |
|---|---|---|
| BERT | [TensorFlow code](./transformers/tf/bert/)<br>[PyTorch code](./transformers/pytorch/bert/) | - |
| BERT (fine-tuning) Classifier | [TensorFlow code](./transformers/tf/bert/fine_tuning/classifier/)<br>[PyTorch code](./transformers/pytorch/bert/fine_tuning/classifier/) | - |
| BERT (fine-tuning) Named Entity Recognition | [TensorFlow code](./transformers/tf/bert/fine_tuning/token_classifier/)<br>[PyTorch code](./transformers/pytorch/bert/fine_tuning/token_classifier/) | - |
| BERT (fine-tuning) Summarization | [TensorFlow code](./transformers/tf/bert/fine_tuning/extractive_summarization/)<br>[PyTorch code](./transformers/pytorch/bert/fine_tuning/extractive_summarization/) | - |
| BERT (fine-tuning) Question Answering | [TensorFlow code](./transformers/tf/bert/fine_tuning/qa/)<br>[PyTorch code](./transformers/pytorch/bert/fine_tuning/qa/) | - |
| GPT-2 | [TensorFlow code](./transformers/tf/gpt2/)<br>[PyTorch code](./transformers/pytorch/gpt2/) | [TensorFlow code](./transformers/tf/gpt2/) |
| GPT-3 | - | [TensorFlow code](./transformers/tf/gpt2/) |
| GPT-J | - | [TensorFlow code](./transformers/tf/gptj/) |
| GPT-J (fine-tuning) Summarization | - | [TensorFlow code](./transformers/tf/gptj/fine_tuning/abstractive_summarization/) |
| Linformer | [TensorFlow code](./transformers/tf/linformer/) | - |
| RoBERTa | [TensorFlow code](./transformers/tf/bert/)<br>[PyTorch code](./transformers/pytorch/bert/) | - |
| T5 | [TensorFlow code](./transformers/tf/t5/)<br>[PyTorch code](./transformers/pytorch/t5/) | - |
| Transformer | [TensorFlow code](./transformers/tf/transformer/)<br>[PyTorch code](./transformers/pytorch/t5/) | - |
| MNIST (fully connected) | [TensorFlow code](./fc_mnist/tf/)<br>[PyTorch code](./fc_mnist/pytorch/) | - |
| 2D UNet (experimental) | [TensorFlow code](./unet/tf/) | - |

## License

[Apache License 2.0](./LICENSE)

***

### Footnotes

<sub>\* Only supported in the the Layer Pipelined mode. Not supported in Weight Streaming mode yet.</sub>
