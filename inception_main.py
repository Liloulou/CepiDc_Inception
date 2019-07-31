import tensorflow as tf
from inception import pipe
from inception.model import inception
from utils import metrics

INF = int(1e9)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):

    with tf.name_scope('learning_rate'):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)

        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        tf.identity(learning_rate, 'learning_rate')

        return learning_rate


def get_train_op_and_metrics(loss, params):

    with tf.variable_scope('get_train_op'):
        learning_rate = get_learning_rate(
            params['learning_rate'],
            params['hidden_size'],
            params['learning_rate_warmup_steps']
        )

        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params['optimizer_adam_beta1'],
            beta2=params['optimizer_adam_beta2'],
            epsilon=params['optimizer_adam_epsilon']
        )

        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        clipped_gradients = tf.clip_by_global_norm([grad[0] for grad in gradients], params['gradient_clipping'])[0]
        minimize_op = optimizer.apply_gradients(
            [(grad, var[1]) for grad, var in zip(clipped_gradients, gradients)], global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        return train_op


def model_fn(features, labels, mode, params):
    code_table = tf.contrib.lookup.index_table_from_file(
        params['icd_file'], vocab_size=params['vocab_size'])

    inputs, targets = pipe.input_layers(
        features, labels, code_table, pipe.make_columns(params)
    )

    initializer = tf.variance_scaling_initializer(
        params["initializer_gain"], mode="fan_avg", distribution="uniform")

    model = inception.Model(params, mode == tf.estimator.ModeKeys.TRAIN, initializer)

    logits = model(inputs['codes'], inputs['aux_inputs'])
    loss = metrics.cross_entropy_loss(logits, targets, params['label_smoothing'], params['vocab_size'])

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics_dict = {
            'accuracy': tf.metrics.accuracy(
                targets,
                tf.argmax(logits, axis=-1)
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, predictions={'predictions': logits},
            eval_metric_ops=metrics_dict
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op, metric_dict = get_train_op_and_metrics(loss, params)

        metric_dict['minibatch_loss'] = loss

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


"""
a = tf.contrib.distribute.MirroredStrategy(
    num_gpus=2,
    cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
        all_reduce_alg="hierarchical_copy"
    )
)"""