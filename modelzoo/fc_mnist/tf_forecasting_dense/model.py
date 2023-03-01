import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import Policy

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec

from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.SquaredErrorLayer import SquaredErrorLayer

def build_model(features, labels, mode, params):
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')

    mixed_precision = params["model"]["mixed_precision"]
    boundary_casting = params["model"]["boundary_casting"]
    tf_summary = params["model"]["tf_summary"]

    batch_size = params["train_input"]["batch_size"]

    seq_len_2 = params["train_input"]["dim_half"]-params["train_input"]["enc_end"]

    # (batch_size, 2475) -> (batch_size, 90)
    dense_layer = DenseLayer(seq_len_2, use_bias=True, trainable=True, dtype=dtype, boundary_casting=boundary_casting, tf_summary=tf_summary)
    encoder_input = features["encoder_input"]
    outputs = dense_layer(encoder_input)

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
