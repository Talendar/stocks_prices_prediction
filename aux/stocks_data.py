""" Implements a custom class to help managing the stocks data.

@author: Gabriel Nogueira (Talendar)
"""

import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class StocksData:
    def __init__(self,
                 data_df,
                 num_sessions,
                 labels_names,
                 feature_normalization=None,  # tuple: (norm_func, denorm_func)
                 label_normalization=None,  # tuple: (norm_func, denorm_func)
                 data_split_pc=(.8, .12, .08),
                 batch_size=32):
        self.raw = data_df
        self._num_sessions = num_sessions
        self._labels_names = labels_names
        self._batch_size = batch_size

        # cleaning
        if "adjclose" in self.raw.columns:
            self.raw.drop("adjclose", axis=1, inplace=True)
            
        if "splits" in self.raw.columns:
            self.raw.drop("splits", axis=1, inplace=True)

        # splitting
        self.raw_train = self.raw_val = self.raw_test = None
        self._split_data(data_split_pc)

        # normalization
        self.norm_train = {"features": None, "labels": None}
        self.norm_val = {"features": None, "labels": None}
        self.norm_test = {"features": None, "labels": None}

        if feature_normalization is not None:
            norm_f, self._denorm_features = feature_normalization
            self._norm_features_params = self._normalization(norm_f, "features")

        if label_normalization is not None:
            norm_f, self._denorm_labels = label_normalization
            self._norm_labels_params = self._normalization(norm_f, "labels")

        # datasets
        self.train_ds = self._make_ds(self.raw_train, self.norm_train)
        self.val_ds = self._make_ds(self.raw_val, self.norm_val)
        self.test_ds = self._make_ds(self.raw_test, self.norm_test)

    def raw_data(self, set_name):
        return getattr(self, "raw_" + set_name)
    
    def norm_data(self, set_name):
        return getattr(self, "norm_" + set_name)

    def dataset(self, set_name):
        return getattr(self, set_name + "_ds")
    
    def denormalize_features(self, features):
        return self._denorm_features(features, *self._norm_features_params)
    
    def denormalize_labels(self, labels):
        return self._denorm_labels(labels, *self._norm_labels_params)

    def _split_data(self, pc):
        assert np.sum(pc) == 1, "Data split percentage sum must be 1 (100%)!"
        n = len(self.raw)
        i_trn = int(n*pc[0])
        i_val = i_trn + int(n*pc[1])
        
        # splitting
        train_data, train_labels = self._features_and_labels(
            self.raw[:i_trn])
        val_data, val_labels = self._features_and_labels(
            self.raw[(i_trn - self._num_sessions) : i_val])
        test_data, test_labels = self._features_and_labels(
            self.raw[(i_val - self._num_sessions):])

        # asserting labels
        self._assert_labels(train_data, train_labels)
        self._assert_labels(val_data, val_labels)
        self._assert_labels(test_data, test_labels)

        # dict
        self.raw_train = {"features": train_data, "labels": train_labels}
        self.raw_val = {"features": val_data, "labels": val_labels}
        self.raw_test = {"features": test_data, "labels": test_labels}

    def _features_and_labels(self, data):
        return data[:-self._num_sessions], data[self._labels_names][self._num_sessions:]

    def _assert_labels(self, data, labels):
        data, labels = data[self._labels_names].values, labels.values
        for i in range(len(labels) - self._num_sessions):
            assert np.array_equal(labels[i], data[i + self._num_sessions])

    def _normalization(self, norm, name):
        params = norm(self.raw_train[name])
        self.norm_train[name], params = params[0], params[1:]
        self.norm_val[name] = norm(self.raw_val[name], *params)[0]
        self.norm_test[name] = norm(self.raw_test[name], *params)[0]
        return params

    def _make_ds(self, data, norm_data):
        return timeseries_dataset_from_array(
            data=data["features"] if norm_data["features"] is None 
                                  else norm_data["features"],
            targets=data["labels"] if norm_data["labels"] is None
                                   else norm_data["labels"],
            sequence_length=self._num_sessions,
            sequence_stride=1,
            shuffle=False,
            batch_size=self._batch_size)
        

