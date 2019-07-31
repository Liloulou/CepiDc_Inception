import tensorflow as tf
import tensorflow.contrib as contrib

MAX_RANK = 20
COL_NAMES = ['causeini', 'age', 'age_cat', 'sexe', 'annee'] + ['cim10_' + str(ligne) + '_' + str(rang) for ligne in range(6) for rang in range(MAX_RANK)]
COL_TYPES = [''] + [0.] + [0] * 3 + [''] * 120

vocab_list = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '.'
]

AGE_MEAN = 28103.390003363886
AGE_STD = 6044.013550763499


def _parse_line(line):
    """
    takes in a line of a csv file and returns its data as a feature dictionary
    :param line: the csv file's loaded line
    :return: the associated feature dictionary
    """

    fields = tf.decode_csv(line, record_defaults=COL_TYPES)
    features = dict(zip(COL_NAMES, fields))

    return features


def _pre_process(line):
    """
    Overheads all csv processing functions.
    :param line: a raw csv line
    :return:
    """

    features = _parse_line(line)

    features['sexe'] -= 1

    features['annee'] -= 2000

    labels = features.pop('causeini')

    lines = [tf.stack(
        [features.pop('cim10_' + str(ligne) + '_' + str(rang)) for rang in range(MAX_RANK)],
        axis=0,
    ) for ligne in range(6)]

    features['codes'] = tf.stack(lines, axis=0)

    return features, labels


def csv_input_fn(dataset_name, year, batch_size, num_epochs):
    """
    A predefined input function to feed an Estimator csv based cepidc files
    :param dataset_name: the file's ending type (either 'train, 'valid' or 'test')
    :param batch_size: the size of batches to feed the computational graph
    :param num_epochs: the number of time the entire dataset should be exposed to a gradient descent iteration
    :return: a BatchedDataset as a tuple of a feature dictionary and the labels
    """

    dataset = tf.data.TextLineDataset('data/cepidc_' + str(year) + '_' + dataset_name + '.csv').skip(1)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(num_epochs)
    dataset = dataset.map(lambda x: _pre_process(x))

    return dataset


def multiple_csv_input_fn(dataset_name, batch_size, num_epochs):
    """
    A predefined input function to feed an Estimator csv based cepidc files
    :param dataset_name: the file's ending type (either 'train, 'valid' or 'test')
    :param batch_size: the size of batches to feed the computational graph
    :param num_epochs: the number of time the entire dataset should be exposed to a gradient descent iteration
    :return: a BatchedDataset as a tuple of a feature dictionary and the labels
    """

    filenames = ['data/cepidc_' + str(year) + '_' + dataset_name + '.csv' for year in range(2000, 2016)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(
        lambda filename: (
            tf.data.TextLineDataset(filename).skip(1).map(_pre_process)
        ),
        cycle_length=16,
    )

    # dataset = dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True).repeat(num_epochs) #TODO put that back after testing
    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(num_epochs)

    return dataset


def make_columns(params):

    columns_dict = {}

    columns_dict['codes'] = tf.feature_column.numeric_column(
        key='codes',
        dtype=tf.int32,
        shape=[6, 20]
    )

    columns_dict['age'] = tf.feature_column.numeric_column(
        key='age',
        dtype=tf.float32,
        normalizer_fn=lambda x: (x - AGE_MEAN) / AGE_STD
    )

    columns_dict['age_cat'] = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
            'age_cat',
            num_buckets=26
        ),
        dimension=params['hidden_size']
    )

    columns_dict['sexe'] = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
            'sexe',
            num_buckets=2
        ),
        dimension=params['hidden_size']
    )

    columns_dict['annee'] = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
            'annee',
            num_buckets=16
        ),
        dimension=params['hidden_size'],
    )

    columns_dict['labels'] = tf.feature_column.numeric_column(
        key='labels',
        dtype=tf.int32
    )

    return columns_dict


def input_layers(features, labels, table, feature_columns):

    features_dict = {}

    features['codes'] = table.lookup(features['codes'])

    """features_dict['codes'] = tf.to_int32(tf.feature_column.input_layer(
        features=features,
        feature_columns=feature_columns['codes'],
        trainable=False
    ))"""
    features_dict['codes'] = features['codes']

    aux_inputs = 0
    for key in ['annee', 'age_cat', 'sexe']:
        aux_inputs += tf.feature_column.input_layer(
            features,
            feature_columns[key],
        )

    features_dict['aux_inputs'] = aux_inputs

    labels = table.lookup(labels)

    """labels = tf.to_int32(tf.feature_column.input_layer(
        {'labels': labels},
        feature_columns['labels'],
        trainable=False
    ))"""

    return features_dict, labels
