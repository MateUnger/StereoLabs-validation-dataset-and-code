import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def normalize_vector(vector: np.ndarray):
    """
    Returns the unit vector of the input vector.

    Args:
        vector: input vector

    Returns:
        normalized_vector: unit vector of magnitude 1 with the same direction as the input
    """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Returns the angle in degrees between vectors 'vector_1' and 'vector_2'

    Args:
        vector_1: input vector
        vector_2: other input vector
    Returns:
        angle: angle between the two input vectors in degrees.

    """
    # calculate unit vectors
    vector_1_unit = normalize_vector(vector_1)
    vector_2_unit = normalize_vector(vector_2)

    # get scalar product
    scalar_product = np.clip(np.dot(vector_1_unit, vector_2_unit), -1, 1)

    # clip scalar product to [-1,1] range so trig. function works normal,
    # use arccos to get the angle from scalar product
    angle = np.degrees(np.arccos(scalar_product))

    return angle


def project_vector_on_plane(plane_normal_vector: np.ndarray, vector: np.ndarray):
    """
    Project an n-dimensional vector onto an n-dimensional plane defined by its normal (orthogonal) vector.

    Args:
        plane_normal_vector: vector orthogonal to the reference plane
        vector: vector to be projected
    Returns:
        vector_projection: projection of the input vector onto reference plane
    """
    # normalize surface normal
    normal_vector = plane_normal_vector / np.linalg.norm(plane_normal_vector)
    # multily vector with its scalar product (dot product w surface normal), shift it
    vector_projection = vector - np.dot(vector, normal_vector) * normal_vector
    return vector_projection


def get_projection(plane_normal_vectors: np.ndarray, vector_array: np.ndarray):
    """
    Project each member of an array of vectors onto plane.

    Args:
        plane_normal_vector: vector orthogonal to the reference plane
        vector_array: array of vectros to be projected. Usually a limb for 1 gait-cycle

    Returns:
        array of projected vectors
    """

    return np.array(
        [
            project_vector_on_plane(plane_normal, vector)
            for plane_normal, vector in zip(plane_normal_vectors.T, vector_array.T)
        ]
    )


def compute_asymmetry(left_values: np.ndarray, right_values: np.ndarray) -> float:
    """
    Calculate the asymmetry between means of left and right sides for a given set of values.

    Args:
        left_values: 1D array of values of the left side
        right_values: 1D array of values of the right side

    Returns:
        Percentile difference between smaller and larger means.
        If either of the input arrays are all nans or the larger mean is not positive returns np.nan
    """
    if not (np.isnan(left_values).all() or np.isnan(right_values).all()):
        left_mean = np.nanmean(left_values)
        right_mean = np.nanmean(right_values)
        smaller = np.nanmin([left_mean, right_mean])
        larger = np.nanmax([left_mean, right_mean])
        return 100 * (1 - smaller / larger) if larger > 0 else np.nan
    return np.nan


def load_keypoints_and_labels(npz_data_path: str) -> tuple[np.ndarray, list]:
    """
    Load keypoint and label data from .npz file
    File has to have keys: "keypoints", "labels"

    Args:
        npz_data_path: path to the .npz file

    Returns:
        keypoints: loaded keypoint data as np.array
        labels: loaded labels as list
    """
    if os.path.exists(npz_data_path):
        loaded_data = np.load(npz_data_path)
        keypoints = loaded_data["keypoints"]
        labels = loaded_data["labels"].tolist()
        print(f"loaded data from: {npz_data_path} with keys: {[key for key in loaded_data.keys()]}")
    else:
        raise FileNotFoundError(f"File: {npz_data_path} does not exist!")
    return keypoints, labels


def remove_nan_positions(arr1: np.ndarray, arr2: np.ndarray) -> tuple:
    """
    Remove Nan elements from both input arrays (of equal length).
    Elements are only kepth if at position P both arrays have valid values.

    Args:
        arr1: first input array
        arr2: second input array

    Returns:
        arr1_cleaned: arr1 containing elements where both arr1 and arr2 are valid (not Nan)
        arr2_cleaned: arr2 ....same as above

    """
    if arr1.shape != arr2.shape:
        raise Exception(
            f"Input arrays must have the same shape. array_1: {arr1.shape}, array2: {arr2.shape}"
        )

    # Find positions of NaNs in both arrays
    nan_positions_arr1 = np.isnan(arr1)
    nan_positions_arr2 = np.isnan(arr2)

    # Combine positions to find indices to remove
    nan_positions_combined = nan_positions_arr1 | nan_positions_arr2

    # Filter out the NaN positions from both arrays
    arr1_cleaned = arr1[~nan_positions_combined]
    arr2_cleaned = arr2[~nan_positions_combined]

    return arr1_cleaned, arr2_cleaned


def bland_altman_statistics(method_a: np.ndarray, method_b: np.ndarray, plot=False) -> tuple:
    """
    Calculate Bland-Altman statistics and optionally plot the Bland-Altman plot.

    Args:
        method_a: Array-like, measurements from method A.
        method_b: Array-like, measurements from method B.
        plot: Boolean, if True, plots the Bland-Altman plot.

    Returns:
        bias: Mean difference between the methods.
        rpc: Reproducibility coefficient (1.96 * standard deviation of differences).
        cv: Coefficient of variation.
    """

    # Calculating differences and means
    differences = method_b - method_a
    mean_difference = np.mean(differences)
    std_dev_difference = np.std(differences, ddof=1)
    means = (method_a + method_b) / 2

    # Bias (Mean Difference)
    bias = mean_difference

    # Reproducibility Coefficient (RPC)
    rpc = 1.96 * std_dev_difference

    # Coefficient of Variation (CV)
    cv = (std_dev_difference / abs(bias)) * 100

    # Bland-Altman Plot
    if plot:
        plt.scatter(means, differences)
        plt.axhline(mean_difference, color="gray", linestyle="--")
        plt.axhline(mean_difference + rpc, color="gray", linestyle="--")
        plt.axhline(mean_difference - rpc, color="gray", linestyle="--")
        plt.title("Bland-Altman Plot")
        plt.xlabel("Mean of Two Methods")
        plt.ylabel("Difference Between Methods")
        plt.show()

        # Printing the calculated values
        print(f"Bias (Mean Difference): {bias}")
        print(f"Reproducibility Coefficient (RPC): {rpc}")
        print(f"Coefficient of Variation (CV): {cv:.2f}%")

    return bias, rpc, cv


def uniform_statistics(ground_truth_measurements, new_system_measurements):
    """
    Calculate Pearson correlation coefficient, Bland-Altman statistics, and ICC between two methods.

    Parameters:
    - ground_truth_measurement: Array-like, measurements from method A (ground truth).
    - new_system_measurements:  Array-like, measurements from method B (new measurement system).

    Returns:
    - correlation_coefficient: Pearson correlation coefficient between the two methods.
    - p_value: P-value for the Pearson correlation.
    - bias: Mean difference between the methods.
    - rpc: Reproducibility coefficient.
    - cv: Coefficient of variation.
    - icc_results: DataFrame with ICC results.
    """

    ground_truth_data = np.array(ground_truth_measurements)
    new_system_data = np.array(new_system_measurements)

    # Filter arrays for NaNs
    ground_truth_data, new_system_data = remove_nan_positions(ground_truth_data, new_system_data)

    # absolute error
    absolute_error = np.mean(np.abs(ground_truth_data - new_system_data))
    # relative error
    relative_error = (
        np.mean(np.abs((ground_truth_data - new_system_data) / ground_truth_data)) * 100
    )

    # Calculate RMSE
    rmse = np.mean(np.sqrt(np.mean((ground_truth_data - new_system_data) ** 2)))
    # relative RMSE
    relative_rmse = (rmse / np.mean(ground_truth_data)) * 100

    mean_gt = np.nanmean(ground_truth_data)
    mean_ns = np.nanmean(new_system_data)
    std_gt = np.nanstd(ground_truth_data)
    std_ns = np.nanstd(new_system_data)

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(ground_truth_data, new_system_data)

    # Calculate Bland-Altman statistics
    bias, rpc, cv = bland_altman_statistics(ground_truth_data, new_system_data)

    # Calculate ICC
    icc_results = icc_statistics(ground_truth_data, new_system_data)
    # get the ICC(3,1) result from the table
    icc_3_1 = float(icc_results.loc[icc_results["Type"] == "ICC3"]["ICC"].iloc[0])

    stat_results = {
        "mean_gt": mean_gt,
        "mean_ns": mean_ns,
        "std_gt": std_gt,
        "std_ns": std_ns,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "correlation_coefficient": correlation_coefficient,
        "p_value": p_value,
        "bias": bias,
        "rcp": rpc,
        "cv": cv,
        "icc_3_1": icc_3_1,
    }

    return stat_results


def icc_statistics(method_a: np.ndarray, method_b: np.ndarray) -> pd.DataFrame:
    """
    Calculate Intraclass Correlation Coefficient (ICC) between two methods.

    Args:
        method_a: Array-like, measurements from method A.
        method_b: Array-like, measurements from method B.

    Returns:

        icc_results: DataFrame with ICC results.
    """

    subjects = list(range(1, len(method_a) + 1))

    # Prepare the DataFrame
    data = pd.DataFrame(
        {
            "Subject_ID": subjects + subjects,  # Repeat subject IDs for each method
            "Measurement": list(method_a) + list(method_b),  # Combine measurements
            "Method": ["A"] * len(method_a) + ["B"] * len(method_b),  # Label methods
        }
    )

    # Convert "Method" to categorical
    data["Method"] = pd.Categorical(data["Method"])

    # Calculate ICCs
    icc_results = pg.intraclass_corr(
        data=data, targets="Subject_ID", raters="Method", ratings="Measurement"
    )

    return icc_results
