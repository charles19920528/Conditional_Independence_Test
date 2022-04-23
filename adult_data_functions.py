import numpy as np
import pandas as pd
import generate_train_functions as gt
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score


def preprocess(adult_dt_path, categorical_feature_encoder, sex_encoder, race_encoder, income_encoder,
               encoder_fit_boolean):
    """
    Preprocess the adult data.

    :param adult_dt_path: A string. Path to the adult data to be loaded.
    :param categorical_feature_encoder:
    :param sex_encoder: sklearn.preprocessing.LabelEncode
    :param race_encoder: sklearn.preprocessing.LabelEncode
    :param income_encoder: sklearn.preprocessing.LabelEncode
    :param encoder_fit_boolean: A boolean. If true, the function will fit the encoder on the data.

    :return:
        A dictionary.
    """
    name_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income-label"]

    adult_dt = pd.read_csv(adult_dt_path, names=name_list)
    if adult_dt_path[-4:] == "test":
        adult_dt = adult_dt.loc[adult_dt["sex"].notna()]
        adult_dt["income-label"] = adult_dt["income-label"].apply(lambda x: x[:-1])

    categorical_colnames_list = ["workclass", "education", "marital-status", "occupation", "relationship"]
    continuous_colnames_list = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for col in categorical_colnames_list + ["sex", "race", "income-label"]:
        adult_dt[col] = adult_dt[col].str.strip()

    if encoder_fit_boolean:
        sex_encoder.fit(adult_dt["sex"])
        race_encoder.fit(adult_dt["race"])
        income_encoder.fit(adult_dt["income-label"])
        categorical_feature_encoder.fit(adult_dt[categorical_colnames_list])

    data_dict = {"sex": sex_encoder.transform(adult_dt["sex"]),
                 "race": race_encoder.transform(adult_dt["race"]),
                 "income-label": income_encoder.transform(adult_dt["income-label"]),
                 "categorical-features": categorical_feature_encoder.transform(adult_dt[categorical_colnames_list]),
                 "continuous-features": adult_dt[continuous_colnames_list]}

    return data_dict


class ModelNetwork(tf.keras.Model):
    # This is the class network we fit on the data.
    def __init__(self, number_forward_layers, hidden_dim, output_dim=2, education_dim=None,
                 occupation_dim=None):
        super(ModelNetwork, self).__init__()

        self.number_forward_layers = number_forward_layers
        self.categorical_dim = 53
        if education_dim is not None:
            self.education_embedding_layer = tf.keras.layers.Dense(
                units=education_dim,
                input_dim=(16,)
            )
            self.categorical_dim = self.categorical_dim - 16 + education_dim
        self.education_dim = education_dim

        if occupation_dim is not None:
            self.occupation_embedding_layer = tf.keras.layers.Dense(
                units=hidden_dim,
                input_dim=(15,)
            )
            self.categorical_dim = self.categorical_dim - 15 + education_dim
        self.occupation_dim = occupation_dim

        self.initial_block = tf.keras.layers.Dense(
            units=hidden_dim,
            activation="elu"
        )

        if number_forward_layers > 1:
            self.feed_forward_rest_vet = [tf.keras.layers.Dense(
                units=hidden_dim, input_dim=(hidden_dim,), activation="elu"
            ) for _ in np.arange(number_forward_layers - 1)]

        self.final_linear = tf.keras.layers.Dense(
            units=output_dim,
            input_dim=(hidden_dim,)
        )

        self.final_linear_input_mat = None

    def call(self, inputs):
        continuous_tensor, categorical_tensor = inputs
        continuous_tensor = tf.cast(continuous_tensor, tf.float32)
        categorical_tensor = tf.cast(categorical_tensor, tf.float32)
        if len(continuous_tensor.shape) == 1:
            continuous_tensor = tf.reshape(continuous_tensor, (1, -1))
            categorical_tensor = tf.reshape(categorical_tensor, (1, -1))

        # Process Categorical input
        embedding_boolean_edu_array = np.repeat(True, categorical_tensor.shape[1])
        embedding_boolean_occ_array = np.repeat(True, categorical_tensor.shape[1])
        if self.education_dim is not None:
            embedding_boolean_edu_array[9:25] = False
            education_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_edu_array, 1)
            embedded_education_tensor = self.education_embedding_layer(education_tensor)
        if self.occupation_dim is not None:
            embedding_boolean_occ_array[32:47] = False
            occupation_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_occ_array, 1)
            embedded_occupation_tensor = self.occupation_embedding_layer(occupation_tensor)

        not_embedding_boolean_array = embedding_boolean_edu_array & embedding_boolean_occ_array
        categorical_tensor = tf.boolean_mask(categorical_tensor, not_embedding_boolean_array, 1)
        if self.education_dim is not None:
            categorical_tensor = tf.concat([categorical_tensor, embedded_education_tensor], 1)
        if self.occupation_dim is not None:
            categorical_tensor = tf.concat([categorical_tensor, embedded_occupation_tensor], 1)

        input_tensor = tf.concat([continuous_tensor, categorical_tensor], 1)

        output = self.initial_block(input_tensor)
        if self.number_forward_layers != 1:
            for i in np.arange(self.number_forward_layers - 1):
                output = self.feed_forward_rest_vet[i](output)
        self.final_linear_input_mat = output.numpy()
        output = self.final_linear(output)

        return output

    def initialize(self):
        input_tuple = (np.zeros((1, 6)), np.zeros((1, self.categorical_dim)))
        test = self.call(input_tuple)
        return test


def ising_likelihood(y_true, y_pred):
    return gt.log_ising_likelihood(x_y_mat=y_true, parameter_mat=y_pred, reduce_boolean=True)


class Metric(tf.keras.metrics.Metric):
    def __init__(self, name, response_name, **kwargs):
        assert name in {"f1", "accuracy"}, "Possible names are 'f1' and 'accuracy'."
        assert response_name in {"sex", "income-label"}, "Possible response names are 'sex', 'income-label'."
        self.response_name = response_name
        super().__init__(name=f"{name}_{response_name}", **kwargs)
        self.score = self.add_weight(name=f"{name}_{response_name}", initializer="zeros")
        if name == "f1":
            self.metric = f1_score
        else:
            self.metric = accuracy_score

    def update_state(self, y_true, y_pred, sample_weight=None):
        prob_pred = gt.pmf_collection(parameter_mat=y_pred)
        prob_pred = tf.reshape(prob_pred, (-1, 4))

        prob_argmax_tensor = tf.argmax(prob_pred, axis=1)
        x_y_pred_array = np.tile((1, 1), (y_true.shape[0], 1))
        x_y_pred_array[prob_argmax_tensor == 1, :] = [1, -1]
        x_y_pred_array[prob_argmax_tensor == 2, :] = [-1, 1]
        x_y_pred_array[prob_argmax_tensor == 3, :] = [-1, -1]

        i = 1
        if self.response_name == "sex":
            i = 0
        class_score = self.metric(y_true=y_true[:, i], y_pred=x_y_pred_array[:, i], sample_weight=sample_weight)
        self.score.assign(class_score)

    def result(self):
        return self.score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.score.assign(0.0)



# y_true = tf.constant([[1, -1],[1, -1], [1, 1]])
# y_pred = tf.constant([[0.2, 0.1], [1, -1], [-2, 1]])
# metric = Metric(name="accuracy")
# metric.update_state(y_true=y_true, y_pred=y_pred)
# test = metric.result()
# metric.reset_state()
