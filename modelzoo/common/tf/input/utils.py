# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Function for performing standard transformations on datasets.
"""

import numpy as np
import tensorflow as tf


def transform_dataset(
    dataset,
    map_fn,
    batch_size,
    is_training,
    shuffle,
    post_batch_map_fn=None,
    shuffle_buffer=None,
    repeat=True,
    seed=None,
    map_before_batch=False,
    batch_fn=None,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    skip_steps=0,
):
    """
    Apply standard transformations to a dataset:
        - shuffle -> batch -> map -> repeat if map_before_batch is False
        - shuffle -> map -> batch -> repeat if map_before_batch is True

    Batching before mapping is generally faster and the preferred method due to
    vectorization of map fn.

    Note: Mapping before batching may be required if parsing TF records that
    contain `FixedLenSequenceFeature` examples (rather than `FixedLenFeature`)

    :param tf.data.Dataset dataset: Dataset to apply transformations to
    :param func map_fn: Mapping function to be applied after batching data
    :param int batch_size: Batch size for model training
    :param bool shuffle: If True, then shuffle the dataset
    :param int shuffle_buffer: Size of shuffle buffer to sample data from
    :param bool repeat: If True, repeat the dataset
    :param int seed: Seed to use for shuffle randomizer or None
    :param bool map_before_batch: if True, mapping will happen before batching.
    :param tf.Tensor num_parallel_calls: representing the number of batches to compute
           asynchronously in parallel. Default value is `tf.data.experimental.AUTOTUNE` when
           number of parallel calls is set dynamically based on available resources.
    :param int skip_steps: Number of steps to skip the dataset after batching.

    :returns: tf dataset
    """

    if batch_fn is None:
        batch_fn = lambda ds: ds.batch(batch_size, drop_remainder=True)

    if is_training and shuffle:
        if not shuffle_buffer:
            shuffle_buffer = 10 * batch_size
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    if not map_before_batch:
        dataset = batch_fn(dataset)

    if map_fn is not None:
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=num_parallel_calls,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (shuffle and seed is None),
        )

    if map_before_batch:
        dataset = batch_fn(dataset)

    if post_batch_map_fn:
        dataset = dataset.map(
            post_batch_map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not (shuffle and seed is None),
        )

    if is_training and repeat:
        dataset = dataset.repeat()

    dataset = dataset.skip(skip_steps)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_int_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    if values is None:
        values = []
    if isinstance(values, np.ndarray) and values.ndim > 1:
        values = values.reshape(-1)
    if not isinstance(values, list):
        values = values.tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def create_float_feature(values):
    """Returns a float_list from a float / double."""
    if values is None:
        values = []
    if isinstance(values, np.ndarray) and values.ndim > 1:
        values = values.reshape(-1)
    if not isinstance(values, list):
        values = values.tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bucketed_batch(
    dataset,
    element_length_func,
    bucket_boundaries,
    batch_size,
    padded_shapes=None,
    padding_values=None,
    no_padding=False,
    drop_remainder=False,
):
    """
    Batch the dataset such that samples within a batch have similar values of
    `features[key]`. Tensorflow has a native implemenation of bucketing
    starting in version 2.6 (see `tf.data.Dataset.bucket_by_sequence_length`).
    This function is intended to provide a subset of the functionality of the
    tensorflow version until tf >= 2.6 is supported in the toolchain. See the
    tensorflow documentation for a detailed interface description.

    :param tf.data.Dataset dataset: an unbatched dataset.
    :param element_length_func: a function that takes a dataset element and
        returns its length.
    :param list bucket_boundaries: a list of boundaries between buckets.
        Expected to have length one less than the total number of buckets.
    :param list batch_size: the batch size for the resulting data. Note that
        this is different from the tensorflow interface as we require a
        static batch size, so we can't have the option for batch size to vary
        based on bucket.
    :param padded_shapes: Possibly nested structure of tuples and dicts
        describing the desired padding for each dataset element. Only
        required if `no_padding = False`.
    :param padding_values: A possibly nested structure of tuples and dicts
        describing the desired padding values for each dataset element. Only
        required if `no_padding = False`.
    :param bool no_padding: Whether or not to pad samples before batching them.
    :param bool drop_remainder: Whether or not to drop the final incomplete
        batch if the number of dataset elements is not a multiple of the batch
        size.
    :returns: a tf.data.Datset of bucketed batched data.
    """
    assert (
        sorted(bucket_boundaries) == bucket_boundaries
    ), "Bucket boundaries must be sorted"
    assert (
        bucket_boundaries[0] > 0
    ), f"Bucket boundaries must be greater than zero, got {bucket_boundaries}."
    buckets_tensor = tf.constant([0] + bucket_boundaries)

    def key_func(*x):
        l = element_length_func(*x)
        index = tf.squeeze(tf.argmax(tf.where(l >= buckets_tensor)))
        return tf.cast(index, tf.int64)

    def reduce_func(key, windowed_data):
        if no_padding:
            return windowed_data.batch(
                batch_size, drop_remainder=drop_remainder
            )
        else:
            return windowed_data.padded_batch(
                batch_size,
                padded_shapes=padded_shapes,
                padding_values=padding_values,
                drop_remainder=drop_remainder,
            )

    return dataset.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size,
        )
    )
