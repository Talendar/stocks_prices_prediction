""" Implements a custom class to help managing the stocks data.

@author: Gabriel Nogueira (Talendar)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class StocksData:
    """ Handles a dataset with a single stock (symbol). """
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
  

class MultiStocksDataset:
    """ Handles multiple datasets of different stocks. """

    def __init__(self, 
                 stocks, 
                 num_sessions,
                 labels_names,
                 batch_size,
                 data_split_pc,
                 feature_normalization=None, 
                 label_normalization=None):
        """ Creates an instance of a multi-dataset.
        
        :param stocks: dictionary with the stocks names as keys and their dataframe
        as values.
        """
        self._num_sessions = num_sessions
        self._batch_size = batch_size

        batch_size_per_ds = int(batch_size / len(stocks))
        self._stocks_data = {
            name: StocksData(data_df=data,
                             num_sessions=num_sessions,
                             labels_names=labels_names,
                             feature_normalization=feature_normalization, 
                             label_normalization=label_normalization,    
                             data_split_pc=data_split_pc,
                             batch_size=batch_size_per_ds) \
            for name, data in stocks.items()
        }
        
        self._size = {
            "train": np.sum( [len(sd.raw_train["features"]) for n, sd in self.stocks] ), 
            "val": np.sum( [len(sd.raw_val["features"]) for n, sd in self.stocks] ), 
            "test": np.sum( [len(sd.raw_test["features"]) for n, sd in self.stocks] ), 
        }
        
        num_features = len(list(stocks.values())[0].columns)
        self._out_sign = (
            tf.TensorSpec(shape=(None, self._num_sessions, num_features), dtype=tf.float64),
            tf.TensorSpec(shape=(None, len(labels_names)), dtype=tf.float64),
        )

        self._tf_datasets = {
            mode: self._MultiDatasetWrapperTF(self, mode)._tf_ds \
            for mode in ["train", "val", "test"]
        }
    
    @property
    def tf_datasets(self):
        return self._tf_datasets

    @property
    def size(self):
        return self._size
    
    @property
    def stocks(self):
        return self._stocks_data.items()

    @property
    def num_batches(self):
        return self._num_batches

    def symbols(self):
        return self._stocks_data.keys

    def __getitem__(self, key):
        return self._stocks_data[key]

    class _MultiDatasetWrapperTF:
        def __init__(self, root, mode):
            self._root = root
            self._mode = mode
            self._iters = None
            self._make_iters()
            self._tf_ds = tf.data.Dataset.from_generator(
                self._generator, output_signature=root._out_sign)

        def _make_iters(self):
            self._iters = {
                name: iter(getattr(sd, f"{self._mode}_ds")) \
                for name, sd in self._root._stocks_data.items()
            }
        
        def __next__(self):
            """ Generates a batch with items from the multi-dataset. """
            inputs, outputs = [], []
            for name, it in self._iters.items():
                if it is not None:
                    try:
                        x, y = next(it)
                    except StopIteration:
                        self._iters[name] = None
                        continue
                    inputs.append(x)
                    outputs.append(y)
            
            if not inputs or not outputs:
                self._make_iters()
                raise StopIteration

            return tf.concat(inputs, axis=0), tf.concat(outputs, axis=0)
        
        def __iter__(self):
            return self

        def _generator(self):
            for x, y in self:
                yield x, y
