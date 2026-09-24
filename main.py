import glob
from data_structures import *

DATA_FOLDER = "dataset"

# WALKING_TYPE = "preferred"
# WALKING_TYPE = "slow"
WALKING_TYPE = "*"

for path in glob.glob(os.path.join(DATA_FOLDER, "*", "*")):
    print(path)
    if False not in [
        os.path.exists(os.path.join(path, file)) for file in ["stereo.npz", "qualisys.npz"]
    ]:
        Pair = RecordingPair(folder_path=path)


# create container to aggregate data
nested_dict = lambda: defaultdict(nested_dict)
aggregated_study_results = nested_dict()
for perspective in ["front", "back", "all"]:
    for parameter in GAIT_PARAMETERS:
        for metric in ["mean", "cv", "asymmetry"]:
            for recording_type in ["qualisys", "stereo"]:
                aggregated_study_results[perspective][parameter][metric][recording_type] = []


# iterate json results for each participant and/or walking speed, fill aggregated container
for path in glob.glob(os.path.join(DATA_FOLDER, "*", WALKING_TYPE, "*stat_results.json")):

    _, participant, walking_speed, _ = path.split("\\")
    with open(path) as json_file:
        pair_results = json.load(json_file)

    for perspective in ["front", "back"]:
        for recording_type in ["qualisys", "stereo"]:
            for parameter in GAIT_PARAMETERS:
                for metric in ["mean", "cv", "asymmetry"]:
                    value = pair_results[perspective][recording_type][parameter][metric]
                    aggregated_study_results[perspective][parameter][metric][recording_type].append(
                        value
                    )
                    aggregated_study_results["all"][parameter][metric][recording_type].append(value)


# container for each perspective
study_results_front = []
study_results_back = []
study_results_all = []

# iterate aggregated data
for perspective in ["front", "back", "all"]:
    for parameter in GAIT_PARAMETERS:
        for metric in ["mean", "cv", "asymmetry"]:
            qualisys_measurements = aggregated_study_results[perspective][parameter][metric][
                "qualisys"
            ]
            stereo_measurements = aggregated_study_results[perspective][parameter][metric]["stereo"]

            # calculate statistics
            stats = uniform_statistics(
                ground_truth_measurements=qualisys_measurements,
                new_system_measurements=stereo_measurements,
            )
            row = [
                parameter,
                metric,
                stats["mean_gt"],
                stats["std_gt"],
                stats["mean_ns"],
                stats["std_ns"],
                stats["rmse"],
                stats["relative_rmse"],
                stats["icc_3_1"],
            ]
            # fill containers
            match perspective:
                case "front":
                    study_results_front.append(row)
                case "back":
                    study_results_back.append(row)
                case "all":
                    study_results_all.append(row)

# create dataframes from containers
df_columns = [
    "PARAMETER",
    "METRIC",
    "Q avg",
    "Q std",
    "S avg",
    "S std",
    "RMSE abs",
    "RMSE rel",
    "ICC",
]
pd.options.display.float_format = "{:,.1f}".format
front = pd.DataFrame(study_results_front, columns=df_columns)
back = pd.DataFrame(study_results_back, columns=df_columns)
all = pd.DataFrame(study_results_back, columns=df_columns)

# save dataframes as .csv
front.to_csv(
    f"{walking_speed}_front.csv" if WALKING_TYPE != "*" else "all_front.csv",
    header=True,
    index=False,
)
back.to_csv(
    f"{walking_speed}_back.csv" if WALKING_TYPE != "*" else "all_back.csv", header=True, index=False
)
all.to_csv(
    f"{walking_speed}_all.csv" if WALKING_TYPE != "*" else "all.csv", header=True, index=False
)
