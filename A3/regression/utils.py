def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    import os
    import numpy as np
    import pandas as pd
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)