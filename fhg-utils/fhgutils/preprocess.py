# --- third-party ---
import numpy as np


def filter_labels(labels, Xt, Xc, y):
    # get labels per window
    y_labels = np.empty([len(y)])
    for i in range(len(y)):
        values, counts = np.unique(y[i], return_counts=True)
        idx = np.argmax(counts)
        if counts[idx] > 0.5 * np.sum(counts):
            # take label only if it occurs more than 50% of the window
            y_labels[i] = int(values[idx])
        else:
            # else schedule it for removal
            y_labels[i] = -1
    # get arrays to filter, and make mask
    y_array = np.asarray(y)
    Xt_array = np.asarray(Xt)
    Xc_array = np.asarray(Xc)
    mask = np.isin(y_labels, labels, invert=True)  # mask are only the values that we do consider
    y_labels = y_labels[mask]
    # filter
    y_filtered = y_array[mask]
    Xt_filtered = Xt_array[mask]
    Xc_filtered = Xc_array[mask]
    print("[INFO] original Xt:", len(Xt), Xt[0].shape)
    print("[INFO] filtered Xt:", len(Xt_filtered), Xt_filtered[0].shape)
    return Xt_filtered, Xc_filtered, y_filtered


def one_label_per_window(y):
    # get labels per window
    y_labels = []
    for i in range(len(y)):
        values, counts = np.unique(y[i], return_counts=True)
        idx = np.argmax(counts)
        # take label only if it occurs more than 50% of the window
        if counts[idx] > 0.5 * np.sum(counts):
            y_labels.append(int(values[idx]))

    print("flattened %i labels: %s" % (len(np.unique(y_labels)), str(np.unique(y_labels))))
    return y_labels


def summarize_labels(y, labelsummary=None):
    y_summarized = np.array(y, copy=True)
    y_original = np.array(y, copy=True)
    if labelsummary is not None:
        for key in labelsummary:
            for label in labelsummary[key]:
                mask = np.ma.masked_equal(y_original, label)
                y_summarized[mask.mask] = key
        print("[INFO] Summarized labels from", np.unique(y_original), "to", np.unique(y_summarized))
    return y_summarized
