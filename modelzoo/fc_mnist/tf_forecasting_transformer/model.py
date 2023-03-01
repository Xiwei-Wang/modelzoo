import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import Policy

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec

from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.SquaredErrorLayer import SquaredErrorLayer

from modelzoo.common.tf.layers.PositionEmbeddingLayer import PositionEmbeddingLayer
from modelzoo.transformers.tf.layers.Encoder import Encoder
from modelzoo.transformers.tf.layers.Decoder import Decoder
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer

from modelzoo.common.tf.layers.ReshapeLayer import ReshapeLayer

from modelzoo.transformers.tf.transformer_utils import (
    create_autoregressive_attention_mask,
)

def build_model(features, labels, mode, params):
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')

    # mixed_precision = params["model"]["mixed_precision"]
    boundary_casting = params["model"]["boundary_casting"]
    tf_summary = params["model"]["tf_summary"]

    batch_size = params["train_input"]["batch_size"]

    embed_dim = params["model"]["embed_dim"] # 128
    dense_dim = params["model"]["dense_dim"] # 64
    num_heads = params["model"]["num_heads"] # 8

    seq_len_1 = params["train_input"]["enc_end"]-params["train_input"]["enc_start"]
    seq_len_2 = params["train_input"]["dim_half"]-params["train_input"]["enc_end"]

    # dense_layer = DenseLayer(seq_len_2, use_bias=True, trainable=True, dtype=dtype, boundary_casting=boundary_casting, tf_summary=tf_summary)
    # encoder_input = features["encoder_input"]
    # outputs = dense_layer(encoder_input)

    positional_enbedding_layer_1 = PositionEmbeddingLayer(max_position_embeddings=seq_len_1, embedding_type="fixed", boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=dtype)
    positional_enbedding_layer_2 = PositionEmbeddingLayer(max_position_embeddings=seq_len_2, embedding_type="fixed", boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=dtype)

    encoder = Encoder(hidden_size=embed_dim, num_heads=num_heads, num_hidden_layers=1, filter_size=dense_dim, boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=dtype)
    decoder = Decoder(hidden_size=embed_dim, num_heads=num_heads, num_hidden_layers=1, filter_size=dense_dim, boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=dtype)

    # conv1dlayer = Conv2DLayer(filters=1, kernel_size=(1, embed_dim), padding='valid', activation="relu", data_format="channels_first", boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=dtype)

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    encoder_input = features["encoder_input"]
    decoder_input = features["decoder_input"]

    batch_size_local = encoder_input.shape[0]

    # encoder_embedding_input = tf.zeros([1, seq_len_1, embed_dim], dtype=tf.float16)
    # decoder_embedding_input = tf.zeros([1, seq_len_2, embed_dim], dtype=tf.float16)
    # encoder_input_embedding = positional_enbedding_layer_1(encoder_embedding_input)
    # decoder_input_embedding = positional_enbedding_layer_2(decoder_embedding_input)
    # encoder_input_embedding = tf.repeat(encoder_input_embedding, batch_size_local, axis = 0)
    # decoder_input_embedding = tf.repeat(decoder_input_embedding, batch_size_local, axis = 0)
    # encoder_input_embedding = tf.concat([encoder_input, encoder_input_embedding[:, :, 1:]], axis = -1)
    # decoder_input_embedding = tf.concat([decoder_input, decoder_input_embedding[:, :, 1:]], axis = -1)

    # encoder_input_embedding = tf.repeat(encoder_input, embed_dim, axis = -1)
    # decoder_input_embedding = tf.repeat(decoder_input, embed_dim, axis = -1)

    encoder_input_embedding = positional_enbedding_layer_1(encoder_input)
    decoder_input_embedding = positional_enbedding_layer_2(decoder_input)

    encoder_output = encoder(encoder_input_embedding, training=is_training)
    attn_autoregressive_mask = create_autoregressive_attention_mask(batch_size=tf.shape(decoder_input_embedding)[0], max_sequence_length=tf.shape(decoder_input_embedding)[1], dtype=decoder_input_embedding.dtype)
    decoder_output = decoder(inputs=decoder_input_embedding, encoder_output=encoder_output, self_attention_mask=attn_autoregressive_mask, training=is_training)
    decoder_output = decoder_output[0]

    # decoder_output = decoder_output[:, tf.newaxis, :, :]
    # output = conv1dlayer(decoder_output)
    # output = tf.squeeze(output, axis = -1)
    # output = tf.squeeze(output, axis = 1)
    # outputs = output

    # (batch_size, 90, 128) -> (batch_size, 90, 1)

    # dense_layer = DenseLayer(1, use_bias=True, trainable=True, dtype=dtype, boundary_casting=boundary_casting, tf_summary=tf_summary)
    # output = dense_layer(decoder_output)
    # output = tf.squeeze(output, axis = -1)
    # outputs = output

    # reshape_layer = ReshapeLayer((seq_len_2,), boundary_casting=boundary_casting, tf_summary=tf_summary)
    # dense_layer = DenseLayer(1, use_bias=True, trainable=True, dtype=dtype, boundary_casting=boundary_casting, tf_summary=tf_summary)
    # output = dense_layer(decoder_output)
    # output = reshape_layer(output)
    # outputs = output

    reshape_layer = ReshapeLayer((seq_len_2,), boundary_casting=boundary_casting, tf_summary=tf_summary)
    dense_layer = DenseLayer(1, use_bias=False, trainable=True, dtype=dtype, boundary_casting=boundary_casting, tf_summary=tf_summary)
    output = dense_layer(decoder_output)
    output = reshape_layer(output)
    outputs = output

    loss_layer = SquaredErrorLayer(boundary_casting=boundary_casting, tf_summary=tf_summary, dtype=tf.float32)

    loss = loss_layer(labels, outputs)
    loss = tf.reduce_sum(loss)
    loss = loss / batch_size
    tf.compat.v1.summary.scalar('loss', loss)
    return loss, outputs

def model_fn(features, labels, mode, params):
    loss, outputs = build_model(features, labels, mode, params)

    train_op = None
    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer_params = params["optimizer"]

        optimizer_type = optimizer_params.get("optimizer_type", None)
        if optimizer_type is None or optimizer_type.lower() == "adam":
            opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate=optimizer_params['learning_rate'],
                beta1=optimizer_params['beta1'],
                beta2=optimizer_params['beta2'],
                epsilon=optimizer_params['epsilon'],
            )
        elif optimizer_type.lower() == "sgd":
            opt = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=optimizer_params["learning_rate"]
            )
        else:
            raise ValueError(f'Unsupported optimizer {optimizer_type}')

        train_op = opt.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_or_create_global_step(),
        )
    elif mode == tf.estimator.ModeKeys.EVAL:

        def build_eval_metric_ops(outputs, labels):
            return {
                "mse": tf.compat.v1.metrics.mean_squared_error(labels, outputs),
            }

        host_call = (build_eval_metric_ops, [outputs, labels])
    else:
        raise ValueError("Only TRAIN and EVAL modes supported")

    espec = CSEstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, host_call=host_call,
    )

    return espec
