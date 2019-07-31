# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        """Specify characteristic parameters of embedding layer.

        Args:
        vocab_size: Number of tokens in the embedding. (Typically ~32,000)
        hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            self.shared_weights = tf.get_variable(
                "weights", [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    0., self.hidden_size ** -0.5
                )
            )

        self.built = True

    def call(self, x):
        """Get token embeddings of x.

        Args:
          x: An int32 tensor with shape [batch_size, num_lines, num_ranks]
        Returns:
          embeddings: float32 tensor with shape [batch_size, num_lines, num_ranks, embedding_size]
        """
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, num_lines, num_ranks]
            mask = tf.to_float(tf.not_equal(x, -1))
            x_parse = x * tf.to_int64(mask)
            embeddings = tf.gather(self.shared_weights, x_parse)
            embeddings *= tf.expand_dims(mask, -1)

        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.hidden_size ** 0.5

        return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
          x: A float32 tensor with shape [batch_size, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, vocab_size].
        """

        with tf.name_scope("presoftmax_linear"):

            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return logits
