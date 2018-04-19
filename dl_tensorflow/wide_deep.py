import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], ['']]
NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


class Config(object):
    """配置参数
    
    Arguments:
        object {[type]} -- [description]
    """

    data_dir = "./data/adult"
    epochs = 40
    between_evals = 2
    batch_size = 40


def build_columns():
    """定义将使用的基础类别型和连续型特征列
    """

    # 连续型特征列
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")
    # 教育背景
    education = tf.feature_column.categorical_column_with_vocabulary_list("education", [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
    ])
    # 婚姻状况
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list("marital_status", [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'
    ])
    # 关系
    relationship = tf.feature_column.categorical_column_with_vocabulary_list("relationship", [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'
    ])
    # 工作类型
    workclass = tf.feature_column.categorical_column_with_vocabulary_list("workclass", [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'
    ])
    # 哈希的例子
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",
                                                                       hash_bucket_size=1000)
    # 转换,将可能值的范围分成子界,将一个值变成这个值的绝对标签。根据年龄进行分类
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # 宽度模型是一个线性模型，具有一系列稀疏和交叉的特征列：
    base_columns = [
        education, marital_status, relationship, workclass, occupation, age_buckets
    ]
    # 交叉特征列的宽度模型可以有效记住特征之间的稀疏交互，限制是它们不能推广到没有出现在训练数据中的特征组合
    crossed_columns = [
        tf.feature_column.crossed_column(['education', 'occupation'],
                                         hash_bucket_size=1000
                                         ),
        tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'],
                                         hash_bucket_size=1000)
    ]
    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        # indicator_column 来创建一些类别列的 multi-hot 表示
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # embedding_column 为类别列配置嵌入
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def build_model(model_type: str, opt: Config):
    """构建模型
    """

    wide_columns, deep_columns = build_columns()
    model = None
    if model_type == "wide":

        model = tf.estimator.LinearClassifier(
            model_dir="./models/wide",
            feature_columns=wide_columns
        )
    elif model_type == "deep":

        model = tf.estimator.DNNClassifier(
            model_dir="./models/deep",
            feature_columns=deep_columns
        )
    else:
        # 宽度模型和深度模型通过将它们的最终输出的对数似然的和作为预测，然后将预测结果提供给对数损失函数
        model = tf.estimator.DNNLinearCombinedClassifier(
            model_dir="./models/wide_deep",
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50]
        )

    # loss = tf.losses.Reduction.SUM
    # optimizer = tf.train.FtrlOptimizer(learning_rate=1e-3)
    # train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.get_global_step())
    # return tf.estimator.EstimatorSpec(mode=model, loss=loss, train_op=train_op)

    return model


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """
    输入函数
    """
    assert tf.gfile.Exists(data_file), ('%s 文件没找到' % data_file)

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def train():
    """训练模型

    Returns:
        [type] -- [description]
    """
    opt = Config()
    train_file = './data/adult/adult.data'
    test_file = './data/adult/adult.test'

    def train_input_fn():
        return input_fn(train_file,
                        opt.between_evals, True, opt.batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, opt.batch_size)

    model = build_model("wide_deep", opt)
    print("开始训练")

    model.train(input_fn=train_input_fn, steps=2000)
    # 评估
    results = model.evaluate(input_fn=eval_input_fn, steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    # for n in range(opt.epochs // opt.between_evals):
    #     model.train(input_fn=train_input_fn)
    #     # 评估
    #     results = model.evaluate(input_fn=eval_input_fn)
    #     print('Results at epoch', (n + 1) * opt.between_evals)
    #     print('-' * 60)

    #     for key in sorted(results):
    #         print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    train()
