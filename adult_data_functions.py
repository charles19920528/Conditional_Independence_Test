import numpy as np
from scipy.sparse import vstack
import generate_train_functions as gt
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import pandas as pd


def preprocess(adult_dt_path, categorical_feature_encoder, sex_encoder, income_encoder,
               encoder_fit_boolean, merge_education_boolean, merge_country_boolean, drop_prop_male_poor=None,
               dropna_boolean=True):
    """
    Preprocess the adult data.

    :param dropna_boolean:
    :param drop_prop_male_poor:
    :param adult_dt_path: A string. Path to the adult data to be loaded.
    :param categorical_feature_encoder:
    :param sex_encoder: sklearn.preprocessing.LabelEncode
    :param income_encoder: sklearn.preprocessing.LabelEncode
    :param encoder_fit_boolean: A boolean. If true, the function will fit the encoder on the data.
    :param merge_education_boolean: A boolean. If true, categories in 'education' will be merged.
    :param merge_country_boolean: A boolean. If true, categories in ''native-country' will be merged.
    :return:
        A dictionary.
    """
    name_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income-label"]

    adult_dt = pd.read_csv(adult_dt_path, names=name_list, na_values=[" ?"])
    if dropna_boolean:
        adult_dt.dropna(inplace=True)
    if adult_dt_path[-4:] == "test":
        adult_dt = adult_dt.loc[adult_dt["sex"].notna()]
        adult_dt["income-label"] = adult_dt["income-label"].apply(lambda x: x[:-1])

    categorical_colnames_list = ["workclass", "education", "occupation", "native-country"]
    continuous_colnames_list = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for col in categorical_colnames_list + ["sex", "race", "income-label"]:
        adult_dt[col] = adult_dt[col].str.strip()

    if merge_education_boolean:
        edu_map_dict = {
            "11th": "pre_hs",
            "10th": "pre_hs",
            "7th-8th": "pre_hs",
            "9th": "pre_hs",
            "12th": "pre_hs",
            "5th-6th": "pre_hs",
            "1st-4th": "pre_hs",
            "Preschool": "pre_hs"
        }
        adult_dt.replace(edu_map_dict, inplace=True)

    if merge_country_boolean:
        country_map_dict = {
            "Holand-Netherlands": "western_eu",
            "Scotland": "western_eu",
            "France": "western_eu",
            "England": "western_eu",
            "Germany": "western_eu",
            "Italy": "western_eu",
            "Ireland": "western_eu",
            "Columbia": "latin_america",
            "Dominican-Republic": "latin_america",
            "Ecuador": "latin_america",
            "El-Salvador": "latin_america",
            "Haiti": "latin_america",
            "Honduras": "latin_america",
            "Jamaica": "latin_america",
            "Nicaragua": "latin_america",
            "Guatemala": "latin_america",
            "Peru": "latin_america",
            "Trinadad&Tobago": "latin_america",
            "Outlying-US(Guam-USVI-etc)": "us_territory",
            "Puerto-Rico": "us_territory",
            "Greece": "southern_eu",
            "Portugal": "southern_eu",
            "Hong": "greater_china",
            "Taiwan": "greater_china",
            "Laos": "southeast_asia",
            "Thailand": "southeast_asia",
            "Vietnam": "southeast_asia",
            "Hungary": "eastern_eu",
            "Poland": "eastern_eu",
            "Yugoslavia": "eastern_eu",
            "Canada": "north_america",
            "United-States": "north_america"
        }
        adult_dt.replace(country_map_dict, inplace=True)

    if encoder_fit_boolean:
        sex_encoder.fit(adult_dt["sex"])
        income_encoder.fit(adult_dt["income-label"])
        categorical_feature_encoder.fit(adult_dt[categorical_colnames_list])

    data_dict = {"sex": sex_encoder.transform(adult_dt["sex"]),
                 "income-label": income_encoder.transform(adult_dt["income-label"]),
                 "categorical-features": categorical_feature_encoder.transform(adult_dt[categorical_colnames_list]),
                 "continuous-features": adult_dt[continuous_colnames_list].to_numpy().astype("float")}
    if drop_prop_male_poor is not None:
        male_poor_boolean = (data_dict["sex"] == 1) & (data_dict["income-label"] == 0)
        drop_boolean = np.repeat(False, sum(male_poor_boolean))
        drop_n_male_poor = np.int(np.floor(drop_prop_male_poor * sum(male_poor_boolean)))
        drop_boolean[:drop_n_male_poor] = True
        np.random.shuffle(drop_boolean)

        train_data_dict = {
            "sex": np.concatenate([data_dict["sex"][male_poor_boolean][~drop_boolean],
                                   data_dict["sex"][~male_poor_boolean]]),
            "income-label": np.concatenate([data_dict["income-label"][male_poor_boolean][~drop_boolean],
                                            data_dict["income-label"][~male_poor_boolean]]),
            "categorical-features": vstack([data_dict["categorical-features"][male_poor_boolean, :][~drop_boolean, :],
                                            data_dict["categorical-features"][~male_poor_boolean, :]]),
            "continuous-features":
                np.concatenate([data_dict["continuous-features"][male_poor_boolean, :][~drop_boolean, :],
                                data_dict["continuous-features"][~male_poor_boolean, :]], axis=0)
        }

        excessive_data_dict = {
            "sex": data_dict["sex"][male_poor_boolean][drop_boolean],
            "income-label": data_dict["income-label"][male_poor_boolean][drop_boolean],
            "categorical-features": data_dict["categorical-features"][male_poor_boolean, :][drop_boolean, :],
            "continuous-features": data_dict["continuous-features"][male_poor_boolean, :][drop_boolean, :]
        }

        return train_data_dict, excessive_data_dict

    return data_dict


#
# class ModelNetwork(tf.keras.Model):
#     # This is the class network we fit on the data.
#     def __init__(self, final_dim, output_dim, n_layers, hidden_dim=None, education_dim=None,
#                  occupation_dim=None, l2=0):
#         """
#
#         :param n_layers: An integer.
#         :param hidden_dim: An integer. Ignore unless number_forward_layers is larger than 1.
#         :param final_dim:
#         :param output_dim:
#         :param education_dim:
#         :param occupation_dim:
#         """
#         super(ModelNetwork, self).__init__()
#
#         self.n_layers = n_layers
#         self.categorical_dim = 53
#         if education_dim is not None:
#             self.education_embedding_layer = tf.keras.layers.Dense(
#                 units=education_dim,
#                 input_dim=(16,)
#             )
#             self.categorical_dim = self.categorical_dim - 16 + education_dim
#         self.education_dim = education_dim
#
#         if occupation_dim is not None:
#             self.occupation_embedding_layer = tf.keras.layers.Dense(
#                 units=occupation_dim,
#                 input_dim=(15,)
#             )
#             self.categorical_dim = self.categorical_dim - 15 + occupation_dim
#         self.occupation_dim = occupation_dim
#
#         self.feed_forward_rest_vet = _create_connnected_block(n_layers=n_layers, hidden_dim=hidden_dim,
#                                                               output_dim=final_dim, l2=l2)
#
#         self.output_dim = output_dim
#         self.final_linear = tf.keras.layers.Dense(
#             units=output_dim,
#             input_dim=(final_dim,),
#             kernel_regularizer=tf.keras.regularizers.l2(l2)
#         )
#
#     def call(self, inputs):
#         continuous_tensor, categorical_tensor = inputs
#         continuous_tensor = tf.cast(continuous_tensor, tf.float32)
#         categorical_tensor = tf.cast(categorical_tensor, tf.float32)
#         if len(continuous_tensor.shape) == 1:
#             continuous_tensor = tf.reshape(continuous_tensor, (1, -1))
#             categorical_tensor = tf.reshape(categorical_tensor, (1, -1))
#
#         # Process Categorical input
#         embedding_boolean_edu_array = np.repeat(True, categorical_tensor.shape[1])
#         embedding_boolean_occ_array = np.repeat(True, categorical_tensor.shape[1])
#         if self.education_dim is not None:
#             embedding_boolean_edu_array[9:25] = False
#             education_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_edu_array, 1)
#             embedded_education_tensor = self.education_embedding_layer(education_tensor)
#         if self.occupation_dim is not None:
#             embedding_boolean_occ_array[32:47] = False
#             occupation_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_occ_array, 1)
#             embedded_occupation_tensor = self.occupation_embedding_layer(occupation_tensor)
#
#         not_embedding_boolean_array = embedding_boolean_edu_array & embedding_boolean_occ_array
#         categorical_tensor = tf.boolean_mask(categorical_tensor, not_embedding_boolean_array, 1)
#         if self.education_dim is not None:
#             categorical_tensor = tf.concat([categorical_tensor, embedded_education_tensor], 1)
#         if self.occupation_dim is not None:
#             categorical_tensor = tf.concat([categorical_tensor, embedded_occupation_tensor], 1)
#
#         input_tensor = tf.concat([continuous_tensor, categorical_tensor], 1)
#         output = input_tensor
#         if self.n_layers != 0:
#             for layer in self.feed_forward_rest_vet:
#                 output = layer(output)
#
#         output = self.final_linear(output)
#
#         return output
#
#     def initialize(self):
#         input_tuple = (np.zeros((1, 6)), np.zeros((1, self.categorical_dim)))
#         test = self.call(input_tuple)
#         return test


def _create_connnected_block(n_layers, hidden_dim, output_dim, l2=0):
    if n_layers == 0:
        return None
    layers_list = []
    for i in range(n_layers):
        if i == n_layers - 1:
            layers_list.append(tf.keras.layers.Dense(units=output_dim, activation="elu",
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))
        else:
            layers_list.append(tf.keras.layers.Dense(units=hidden_dim, activation="elu",
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))

    return layers_list


class ModelNetwork(tf.keras.Model):
    # This is the class network we fit on the data.
    def __init__(self, n_layers, hidden_dim, l_dim, output_dim, l2=0):

        super(ModelNetwork, self).__init__()

        self.hidden_layer = _create_connnected_block(n_layers=n_layers, hidden_dim=hidden_dim,
                                                     output_dim=l_dim,
                                                     l2=l2)

        self.final_linear = tf.keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(l2)
        )

    def call(self, inputs):
        output = inputs
        if self.hidden_layer is not None:
            for layer in self.hidden_layer:
                output = layer(output)
        output = self.final_linear(output)

        return output

@tf.function
def ising_likelihood(y_true, y_pred):
    return gt.log_ising_likelihood(x_y_mat=y_true, parameter_mat=y_pred, reduce_boolean=True) / y_true.shape[0]


def ising_predict(parameter_mat, prob_boolean=True):
    prob_pred = gt.pmf_collection(parameter_mat=parameter_mat)
    prob_pred = tf.reshape(prob_pred, (-1, 4))

    if prob_boolean:
        return prob_pred.numpy()

    prob_argmax_tensor = tf.argmax(prob_pred, axis=1)
    x_y_pred_array = np.tile((1, 1), (parameter_mat.shape[0], 1))
    x_y_pred_array[prob_argmax_tensor == 1, :] = [1, -1]
    x_y_pred_array[prob_argmax_tensor == 2, :] = [-1, 1]
    x_y_pred_array[prob_argmax_tensor == 3, :] = [-1, -1]

    return x_y_pred_array
    # prob_pred = gt.pmf_collection(parameter_mat=parameter_mat)
    # prob_pred = tf.reshape(prob_pred, (-1, 4))
    # prob_x_1_tensor = prob_pred[:, 0] + prob_pred[:, 1]
    # prob_y_1_tensor = prob_pred[:, 0] + prob_pred[:, 2]
    #
    # if not prob_boolean:
    #     x_y_pred_array = np.tile((1, 1), (parameter_mat.shape[0], 1))
    #     x_y_pred_array[prob_x_1_tensor < x_threshold, 0] = -1
    #     x_y_pred_array[prob_y_1_tensor < y_threshold, 1] = -1
    #
    #     return tf.constant(x_y_pred_array)
    #
    # return tf.concat([tf.reshape(prob_x_1_tensor, (-1, 1)), tf.reshape(prob_y_1_tensor, (-1, 1))], axis=1)


# def ising_predict(parameter_mat, x_threshold=0.5, y_threshold=0.5, prob_boolean=True):
#     prob_pred = gt.pmf_collection(parameter_mat=parameter_mat)
#     prob_pred = tf.reshape(prob_pred, (-1, 4))
#
#     prob_argmax_tensor = tf.argmax(prob_pred, axis=1)
#     x_y_pred_array = np.tile((1, 1), (parameter_mat.shape[0], 1))
#     x_y_pred_array[prob_argmax_tensor == 1, :] = [1, -1]
#     x_y_pred_array[prob_argmax_tensor == 2, :] = [-1, 1]
#     x_y_pred_array[prob_argmax_tensor == 3, :] = [-1, -1]
#
#     prob_pred = gt.pmf_collection(parameter_mat=parameter_mat)
#     prob_pred = tf.reshape(prob_pred, (-1, 4))
#     prob_x_1_tensor = prob_pred[:, 0] + prob_pred[:, 1]
#     prob_y_1_tensor = prob_pred[:, 0] + prob_pred[:, 2]
#
#     if not prob_boolean:
#         x_y_pred_array = np.tile((1, 1), (parameter_mat.shape[0], 1))
#         x_y_pred_array[prob_x_1_tensor < x_threshold, 0] = -1
#         x_y_pred_array[prob_y_1_tensor < y_threshold, 1] = -1
#
#         return tf.constant(x_y_pred_array)
#
#     return tf.concat([tf.reshape(prob_x_1_tensor, (-1, 1)), tf.reshape(prob_y_1_tensor, (-1, 1))], axis=1)


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
        """

        :param y_true: An array. Storing the x_y_mat.
        :param y_pred: A tensor. J parameters of the Ising Model
        :param sample_weight:
        :return:
        """
        x_y_pred_array = ising_predict(parameter_mat=y_pred, prob_boolean=False)

        # we made class we want to calculate based on to have label 1.
        if self.response_name == "sex":
            class_score = self.metric(y_true=-y_true[:, 0], y_pred=-x_y_pred_array[:, 0], sample_weight=sample_weight)
        else:
            class_score = self.metric(y_true=y_true[:, 1], y_pred=x_y_pred_array[:, 1], sample_weight=sample_weight)
        self.score.assign(class_score)

    def result(self):
        return self.score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.score.assign(0.0)



def score_summary(y_true, y_pred, pos_label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)

    return pd.DataFrame({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, index=[0])


def sklearn_score_summary(model, feature_mat, y_true, pos_label):
    y_pred = model.predict(feature_mat)

    return score_summary(y_true=y_true, y_pred=y_pred, pos_label=pos_label)


def tf_score_summary(model, dataset,
                     activation=lambda x: ising_predict(x, prob_boolean=False),
                     pos_label_tuple=(-1, 1)):
    output_list = []
    y_true_list = []
    for input, y_true in dataset:
        output_list.append(model(input))
        y_true_list.append(y_true)
    output_tensor = tf.concat(output_list, axis=0)
    if activation is not None:
        output_array = activation(output_tensor)
    y_true_tensor = tf.concat(y_true_list, axis=0)

    sex_score_df = score_summary(y_true=y_true_tensor.numpy()[:, 0],
                                 y_pred=output_array[:, 0], pos_label=pos_label_tuple[0])
    income_score_df = score_summary(y_true=y_true_tensor.numpy()[:, 1],
                                    y_pred=output_array[:, 1], pos_label=pos_label_tuple[1])

    result_df = pd.concat([sex_score_df, income_score_df])
    result_df.index = [f"sex{pos_label_tuple[0]}", f"income{pos_label_tuple[1]}"]

    return result_df
# model = BranchesModel(n_shared_layers=2, shared_hidden_dim=5, shared_output_dim=2, n_x_layers=2, x_hidden_dim=3,
#                       n_y_layers=3, y_hidden_dim=4)
# model = BranchesModel(n_shared_layers=1, shared_hidden_dim=5, shared_output_dim=2, n_x_layers=0, x_hidden_dim=3,
#                       n_y_layers=0, y_hidden_dim=4)
# model.initialize()
# model = ModelNetwork(number_forward_layers=4, hidden_dim=40, final_dim=5, output_dim=2)
# model.initialize()

# y_true = tf.constant([[1, -1],[1, -1], [1, 1]])
# y_pred = tf.constant([[0.2, 0.1], [1, -1], [-2, 1]])
# metric = Metric(name="accuracy")
# metric.update_state(y_true=y_true, y_pred=y_pred)
# test = metric.result()
# metric.reset_state()
# model_args_dict = {"n_layers": 1, "final_dim": 0, "hidden_dim": None,
#                    "education_dim": None, "occupation_dim": None}
# model = ModelNetwork(**model_args_dict)
# model.initialize()


#
# class ModelTest(tf.keras.Model):
#     # This is the class network we fit on the data.
#     def __init__(self, final_dim, output_dim, n_layers, hidden_dim=None, final_layer_regularizer=None):
#         """
#
#         :param n_layers: An integer.
#         :param hidden_dim: An integer. Ignore unless number_forward_layers is larger than 1.
#         :param final_dim:
#         :param output_dim:
#         :param education_dim:
#         :param occupation_dim:
#         """
#         super(ModelTest, self).__init__()
#
#         self.n_layers = n_layers
#
#         self.feed_forward_rest_vet = _create_connnected_block(n_layers=n_layers, hidden_dim=hidden_dim,
#                                                               output_dim=final_dim, regularizer=final_layer_regularizer)
#
#         self.output_dim = output_dim
#         self.final_linear = tf.keras.layers.Dense(
#             units=output_dim,
#             input_dim=(final_dim,)
#         )
#
#     def call(self, inputs):
#         continuous_tensor, categorical_tensor = inputs
#         continuous_tensor = tf.cast(continuous_tensor, tf.float32)
#         categorical_tensor = tf.cast(categorical_tensor, tf.float32)
#         if len(continuous_tensor.shape) == 1:
#             continuous_tensor = tf.reshape(continuous_tensor, (1, -1))
#             categorical_tensor = tf.reshape(categorical_tensor, (1, -1))
#
#         # Process Categorical input
#         embedding_boolean_edu_array = np.repeat(True, categorical_tensor.shape[1])
#         embedding_boolean_occ_array = np.repeat(True, categorical_tensor.shape[1])
#         if self.education_dim is not None:
#             embedding_boolean_edu_array[9:25] = False
#             education_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_edu_array, 1)
#             embedded_education_tensor = self.education_embedding_layer(education_tensor)
#         if self.occupation_dim is not None:
#             embedding_boolean_occ_array[32:47] = False
#             occupation_tensor = tf.boolean_mask(categorical_tensor, ~embedding_boolean_occ_array, 1)
#             embedded_occupation_tensor = self.occupation_embedding_layer(occupation_tensor)
#
#         not_embedding_boolean_array = embedding_boolean_edu_array & embedding_boolean_occ_array
#         categorical_tensor = tf.boolean_mask(categorical_tensor, not_embedding_boolean_array, 1)
#         if self.education_dim is not None:
#             categorical_tensor = tf.concat([categorical_tensor, embedded_education_tensor], 1)
#         if self.occupation_dim is not None:
#             categorical_tensor = tf.concat([categorical_tensor, embedded_occupation_tensor], 1)
#
#         input_tensor = tf.concat([continuous_tensor, categorical_tensor], 1)
#         output = input_tensor
#         if self.n_layers != 0:
#             for layer in self.feed_forward_rest_vet:
#                 output = layer(output)
#
#         output = self.final_linear(output)
#
#         return output
#
#     def initialize(self):
#         input_tuple = (np.zeros((1, 6)), np.zeros((1, self.categorical_dim)))
#         test = self.call(input_tuple)
#         return test
