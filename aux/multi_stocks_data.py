import tensorflow as tf
import numpy as np
from aux.stocks_data import StocksData


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

        self._out_sign = (
            tf.TensorSpec(shape=(None, self._num_sessions, 5), dtype=tf.float64),
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
