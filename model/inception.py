import tensorflow as tf
from inception.model import embedding_layer


class LayerNormalization(tf.layers.Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable('layer_norm_scale', [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable('layer_norm_bias', [self.hidden_size],
                                    initializer=tf.zeros_initializer())

        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, hidden_size, dropout, train):
        self.layer = layer
        self.postprocess_dropout = dropout
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(hidden_size)

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class TemporalBlock(tf.layers.Layer):

    def __init__(self, dilation, params, train, kernel_initializer=None, name=None):
        super(TemporalBlock, self).__init__(name=name)
        self.padding = (params['kernel'] - 1) * dilation
        self.conv_1 = tf.layers.Conv2D(
            filters=params['hidden_size'],
            kernel_size=[1, params['kernel']],
            dilation_rate=[1, dilation],
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer
        )
        self.conv_2 = tf.layers.Conv2D(
            filters=params['hidden_size'],
            kernel_size=[1, params['kernel']],
            dilation_rate=[1, dilation],
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer
        )
        self.layer_norm_1 = LayerNormalization(params['hidden_size'])
        self.layer_norm_2 = LayerNormalization(params['hidden_size'])
        self.dropout = params['layer_postprocess_dropout']
        self.train = train

    def call(self, inputs):
        padded_inputs = tf.pad(inputs, tf.constant([(0, 0), (0, 0), (1, 0), (0, 0)]) * self.padding)
        outputs = self.conv_1(padded_inputs)
        if self.train:
            outputs = tf.nn.dropout(outputs, 1 - self.dropout)
        outputs = self.layer_norm_1(outputs)

        outputs = tf.pad(outputs, tf.constant([(0, 0), (0, 0), (1, 0), (0, 0)]) * self.padding)
        outputs = self.conv_2(outputs)
        if self.train:
            outputs = tf.nn.dropout(outputs, 1 - self.dropout)
        outputs = self.layer_norm_2(outputs)

        return outputs + inputs


class FancyMaxPool(tf.layers.Layer):

    def __init__(self, filters, strides, activation=tf.nn.relu, kernel_initializer=None, name=None):
        super(FancyMaxPool, self).__init__(name=name)

        self.channel_1 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, 3],
                strides=strides,
                padding='valid',
                activation=activation,
                kernel_initializer=kernel_initializer
            )
        ]

        self.channel_2 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, 3],
                strides=strides,
                padding='valid',
                activation=activation,
                kernel_initializer=kernel_initializer
            )
        ]

        self.pool = tf.layers.MaxPooling2D(
            pool_size=[1, 3],
            strides=strides,
            padding='valid'
        )

    def call(self, inputs):
        outputs_1 = inputs
        for i in range(3):
            outputs_1 = self.channel_1[i](outputs_1)
        outputs_2 = inputs
        for i in range(2):
            outputs_2 = self.channel_2[i](outputs_2)
        outputs_3 = self.pool(inputs)

        return tf.concat([outputs_1, outputs_2, outputs_3], axis=-1)


class Inception1(tf.layers.Layer):

    def __init__(self, filters, activation=tf.nn.relu, kernel_initializer=None, name=None):
        super(Inception1, self).__init__(name=name)
        self.channel_1 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            )
        ]
        self.channel_2 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            )
        ]
        self.channel_3 = [
            tf.layers.MaxPooling2D(
                pool_size=2,
                strides=1,
                padding='same'
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
        ]
        self.channel_4 = tf.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=activation,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):

        outputs_1 = inputs
        for layer in self.channel_1:
            outputs_1 = layer(outputs_1)
        outputs_2 = inputs
        for layer in self.channel_2:
            outputs_2 = layer(outputs_2)
        outputs_3 = inputs
        for layer in self.channel_3:
            outputs_3 = layer(outputs_3)
        outputs_4 = self.channel_4(inputs)

        return tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis=-1)


class Inception2(tf.layers.Layer):

    def __init__(self, filters, kernel_size=5, activation=tf.nn.relu, kernel_initializer=None, name=None):
        super(Inception2, self).__init__(name=name)
        self.channel_1 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, kernel_size],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, 1],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, kernel_size],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, 1],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            )
        ]
        self.channel_2 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, kernel_size],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, 1],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            ),
        ]
        self.channel_3 = [
            tf.layers.MaxPooling2D(
                pool_size=2,
                strides=1,
                padding='same'
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
        ]
        self.channel_4 = tf.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=activation,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):

        outputs_1 = inputs
        for layer in self.channel_1:
            outputs_1 = layer(outputs_1)
        outputs_2 = inputs
        for layer in self.channel_2:
            outputs_2 = layer(outputs_2)
        outputs_3 = inputs
        for layer in self.channel_3:
            outputs_3 = layer(outputs_3)
        outputs_4 = self.channel_4(inputs)

        return tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis=-1)


class Inception3(tf.layers.Layer):

    def __init__(self, filters, activation=tf.nn.relu, kernel_initializer=None, name=None):
        super(Inception3, self).__init__(name=name)
        kernel_size = 3
        self.channel_1 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, 1],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, kernel_size],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
        ]
        self.channel_2 = [
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[1, kernel_size],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, 1],
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer
            ),
        ]
        self.channel_3 = [
            tf.layers.MaxPooling2D(
                pool_size=2,
                strides=1,
                padding='same'
            ),
            tf.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
            ),
        ]
        self.channel_4 = tf.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=activation,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        outputs_1 = self.channel_1[0](inputs)
        outputs_1 = self.channel_1[1](outputs_1)
        outputs_1_1 = self.channel_1[2](outputs_1)
        outputs_1_2 = self.channel_1[3](outputs_1)

        outputs_2 = self.channel_2[0](inputs)
        outputs_2_1 = self.channel_2[1](outputs_2)
        outputs_2_2 = self.channel_2[2](outputs_2)

        outputs_3 = inputs
        for layer in self.channel_3:
            outputs_3 = layer(outputs_3)
        outputs_4 = self.channel_4(inputs)

        return tf.concat([outputs_1_1, outputs_1_2, outputs_2_1, outputs_2_2, outputs_3, outputs_4], axis=-1)


class TemporalEncoder(tf.layers.Layer):
    def __init__(self, params, train, kernel_initializer):
        super(TemporalEncoder, self).__init__(name='temporal_encoder')
        self.blocks = []

        for i in range(2):
            self.blocks.append(
                TemporalBlock(2 ** i, params, train, kernel_initializer, name='temporal_block_1')
            )

        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, inputs):

        for block in self.blocks:
            inputs = block(inputs)

        return self.output_normalization(inputs)


class TemporalInception(tf.layers.Layer):
    def __init__(self, params, train, kernel_initializer=None):
        super(TemporalInception, self).__init__(name='temporal_inception')
        self.inception_1 = [PrePostProcessingWrapper(
            Inception1(params['hidden_size'] // 4, kernel_initializer=kernel_initializer, name='inception_1_' + str(i)),
            hidden_size=params['hidden_size'],
            dropout=params['dropout'],
            train=train
        ) for i in range(3)]
        self.inception_2 = [PrePostProcessingWrapper(
            Inception2(params['hidden_size'] // 2, kernel_initializer=kernel_initializer, name='inception_2_' + str(i)),
            hidden_size=params['hidden_size'] * 2,
            dropout=params['dropout'],
            train=train
        ) for i in range(5)]
        self.inception_3 = [PrePostProcessingWrapper(
            Inception3(params['hidden_size'] // 2, kernel_initializer=kernel_initializer, name='inception_3_' + str(i)),
            hidden_size=params['hidden_size'] * 3,
            dropout=params['dropout'],
            train=train
        ) for i in range(2)]

        self.temporal_maxpool_1 = FancyMaxPool(
            filters=params['hidden_size'] // 2, strides=[1, 2]
        )
        self.temporal_maxpool_2 = FancyMaxPool(
            filters=params['hidden_size'] // 2, strides=[1, 2]
        )

        self.layer_norm_1 = LayerNormalization(params['hidden_size'] * 2)
        self.layer_norm_2 = LayerNormalization(params['hidden_size'] * 3)
        self.final_pool = tf.layers.MaxPooling2D(pool_size=[6, 4], strides=[6, 4])
        self.linear = tf.layers.Dense(units=params['hidden_size'], kernel_initializer=kernel_initializer)

    def call(self, inputs):

        for block in self.inception_1:
            inputs = block(inputs)

        inputs = self.temporal_maxpool_1(inputs)
        inputs = self.layer_norm_1(inputs)

        for block in self.inception_2:
            inputs = block(inputs)

        inputs = self.temporal_maxpool_2(inputs)
        inputs = self.layer_norm_2(inputs)

        for block in self.inception_3:
            inputs = block(inputs)

        inputs = self.final_pool(inputs)

        return self.linear(inputs)


class Model(tf.layers.Layer):

    def __init__(self, params, train, kernel_initializer):
        super(Model, self).__init__(self, name='model')
        self.params = params
        self.train = train
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            params['vocab_size'], params['hidden_size']
        )

        self.temporal_encoder = TemporalEncoder(params, train, kernel_initializer)

        self.temporal_inception = TemporalInception(params, train, kernel_initializer)

    def call(self, inputs, aux_inputs):

        embedded_inputs = self.embedding_softmax_layer(inputs)
        encoder_inputs = embedded_inputs + tf.expand_dims(tf.expand_dims(aux_inputs, 1), 1)

        if self.train:
            encoder_inputs = tf.nn.dropout(
                encoder_inputs, 1 - self.params['dropout']
            )

        encoder_out = self.temporal_encoder(encoder_inputs)

        inception_out = tf.squeeze(self.temporal_inception(encoder_out))

        logits = self.embedding_softmax_layer.linear(inception_out)

        return logits


"""
params = model_params.get_params()
print(params['hidden_size'])
a = tf.random.uniform([240, 6, 20], 0, 700, dtype=tf.int64)
aux_inputs = tf.random.uniform([240, 6, 20, params['hidden_size']])

model = Model(params, False)
b = model(a, aux_inputs)

init_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_var)
    test = sess.run(b)
"""
