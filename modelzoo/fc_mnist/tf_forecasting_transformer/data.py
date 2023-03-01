import tensorflow as tf
# import tensorflow_datasets as tfds
import h5py
import math

class generator():
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(self.file_path, 'r')
        self.keys = list(self.f.keys())
        self.x = self.f[self.keys[0]]
        self.x_shape = self.x.shape[1:]
        # print(self.x.shape) # (1580128, 10130) for training
    def __call__(self):
        for i in range(self.x.shape[0]):
            yield self.x[i]
    def shape(self):
        return self.x_shape

def create_dataset(file_path):
    x_generator = generator(file_path)
    x_shape = x_generator.shape()
    x_dataset = tf.data.Dataset.from_generator(x_generator, output_types=tf.float32, output_shapes=x_shape)
    # f = h5py.File(file_path, 'r')
    # keys = list(f.keys())
    # x = f[keys[0]]
    # always return_labels=False
    # y = f[keys[1]]
    # x_dataset = tf.data.Dataset.from_tensor_slices(x)
    # x_chunks = tf.split(x, num_or_size_splits=x_num_chunks)
    # x_datasets = [tf.data.Dataset.from_tensor_slices(x_chunk) for x_chunk in x_chunks]
    # x_dataset = tf.data.Dataset.concatenate(x_datasets)
    # y_dataset = tf.data.Dataset.from_tensor_slices(y)
    # dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    return x_dataset

def preprocess_fn(x, params):
    compute_dtype = (tf.float16 if params["model"]["mixed_precision"] else tf.float32)
    # always n_channels=1
    # print(x.shape) # (10130,)
    x = x[::2]
    # always normailze=False
    enc_start = params["train_input"]["enc_start"] # 5000//2
    enc_end = params["train_input"]["enc_end"] # (10000 - 50)//2
    encoder_input = x[enc_start:enc_end]
    decoder_input = x[enc_end-1:-1]
    # encoder_input = encoder_input[:, tf.newaxis]
    # decoder_input = decoder_input[:, tf.newaxis]
    decoder_target = x[enc_end:]
    features = {}
    features["encoder_input"] = tf.cast(encoder_input, compute_dtype)
    features["decoder_input"] = tf.cast(decoder_input, compute_dtype)
    labels = tf.cast(decoder_target, tf.float32)
    # print(encoder_input.shape, decoder_input.shape, decoder_target.shape) # (2475,) (90,) (90,)
    return features, labels

def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN):
    training = mode == tf.estimator.ModeKeys.TRAIN
    input_params = params["train_input"]
    file_path = input_params["train_path"] if training else input_params["eval_path"]
    
    # x_num_chunks = input_params["train_x_num_chunks"] if training else input_params["eval_x_num_chunks"]

    dataset = create_dataset(file_path)
    
    dataset = dataset.map(
        lambda x: preprocess_fn(x, params),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Cache the preprocessed data in memory to avoid having to preprocess it again
    dataset = dataset.cache()

    # if training and input_params["shuffle"]:
    #     dataset = dataset.shuffle(buffer_size=tf.data.experimental.AUTOTUNE)
    
    batch_size = (
        input_params.get("train_batch_size")
        if training
        else input_params.get("eval_batch_size")
    )
    if batch_size is None:
        batch_size = input_params["batch_size"]
    # print(batch_size) # 32
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if training:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def train_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.TRAIN)

def eval_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.EVAL)
