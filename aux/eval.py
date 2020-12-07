""" Auxiliary functions related to model evaluation.

@author: Gabriel Nogueira (Talendar)
"""

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt


mape = tf.keras.metrics.MeanAbsolutePercentageError()
mae = tf.keras.metrics.MeanAbsoluteError()
mse = tf.keras.metrics.MeanSquaredError()


def eval_plot(predictions, labels, start_date, plot_samples, title):
    if len(labels) != len(predictions):
        labels = labels[:len(predictions)]

    # sampling
    idx = np.random.randint(len(labels.index) - plot_samples) \
          if start_date == "random" \
          else labels.index.index(start_date)

    sl = slice(idx, idx + plot_samples)
    predictions, labels = predictions[sl], labels[sl]

    # plotting
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(top=2.6)
    fig.suptitle(title, fontsize=22, color="#E0E0E0", x=0.52, y=2.76)

    for i, col in enumerate(labels.columns):
        col_title = "Opening Price" if col == "open" else \
                    "Closing Price" if col == "close" else \
                    "Lowest Price"  if col == "low" else \
                    "Highest Price" if col == "high" else col

        # error bars
        bars_errors = [[], []]
        for d in (labels[col] - predictions[col]).values:
            if d > 0:
                bars_errors[1].append(d)
                bars_errors[0].append(0)
            else:
                bars_errors[0].append(-d)
                bars_errors[1].append(0)

        # config
        ax_graph = fig.add_subplot(len(labels.columns), 1, i+1)
        ax_graph.tick_params(axis='both', which='major', labelsize=12, color="#DCDCDC")
        ax_graph.set_title(f"\n{col_title}\n", fontsize=18, color="#E0E0E0")
        ax_graph.set_ylabel('Ibovespa\n', fontsize=16, color="#E0E0E0")

        # plot
        ax_graph.plot(labels[col].index, labels[col].values, 
                marker='o', c='#2ca02c', 
                label='Labels', linewidth=3)
        ax_graph.errorbar(predictions[col].index, predictions[col].values, ecolor="#ff0000",
                    yerr=bars_errors, linestyle="--", elinewidth=2,
                    marker='s', label='Predictions', c='#ff7f0e', 
                    alpha=0.6, linewidth=3)[-1][0].set_linestyle(':')
        ax_graph.legend(fontsize=16)


def eval_print(results):
    print("#"*25)
    for col in results:
        print(
            f"\n[> {col.upper()} <]\n" +
            f"   . MAE:  {results[col]['mae'] : .0f}\n" +
            f"   . MAPE: {results[col]['mape'] : .4f}%\n"
            f"   . MSE:  {results[col]['mse'] : .0f}\n" + 
            f"   . Tendency Accuracy: {results[col]['tend_ac'] : .4f}%\n"
            "\n" + "#" * 25
        )


def eval(model, data, set_name):
    # predictions
    predictions = model.predict(data.dataset(set_name))
    labels = data.raw_data(set_name)["labels"][:len(predictions)]
    predictions = pd.DataFrame(data=predictions, 
                               index=labels.index,
                               columns=labels.columns)
    
    if data.denormalize_labels is not None:
        predictions = data.denormalize_labels(predictions)

    # results
    results = {"general": {"mae": mae(labels, predictions),
                           "mape": mape(labels, predictions),
                           "mse": mse(labels, predictions),
                           "tend_acc": tendency_accuracy(labels, predictions)} }

    for col in labels.columns:
        results[col] = {"mae":  mae(labels[col], predictions[col]),
                        "mape": mape(labels[col], predictions[col]),
                        "mse":  mse(labels[col], predictions[col]),
                        "tend_acc": tendency_accuracy(labels, predictions)}

    return predictions, results


def tendency_accuracy(labels, predictions):
    labels = labels.loc[predictions.index]
    hits = 0
    for i in range(1, len(predictions)):
        l_past = labels.iloc[i - 1]
        l, p = labels.iloc[i], predictions.iloc[i]

        # prices went up
        if l > l_past:
            # true positive
            if p > l_past:
                hits += 1
            # false negative
            else:
                pass
        # prices went down / stayed the same
        else:
            # true negative
            if p <= l_past:
                hits += 1
            # false positive
            else:
                pass

    return hits / (len(predictions) - 1)
