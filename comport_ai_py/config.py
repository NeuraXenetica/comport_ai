# ╔════════════════════════════════════════════════════════════════════╗
# ║   Comport_AI™ is a free open-source HR predictive analytics tool   ║
# ║   that forecasts the likely range of a worker’s future job         ║
# ║   performance. It treats the likely ceiling and likely floor of    ║
# ║   a worker’s future performance as independent entities that are   ║
# ║   modelled by artificial neural networks whose custom loss         ║
# ║   functions enable them to formulate prediction intervals that     ║
# ║   are as small as possible, while being just large enough to       ║
# ║   contain a worker’s actual future performance value in most       ║
# ║   cases.                                                           ║
# ║                                                                    ║
# ║   Developed by Matthew E. Gladden | ©2021-23 NeuraXenetica LLC     ║
# ║   This software is made available for use under                    ║
# ║   GNU General Public License Version 3                             ║
# ║   (please see https://www.gnu.org/licenses/gpl-3.0.html).          ║
# ╚════════════════════════════════════════════════════════════════════╝

"""
This module stores configuration settings and constants and variables 
that are used by multiple modules within the package.
"""

# ======================================================================
# Constants/variables relating to I/O and files.
# ======================================================================

# These values will be calculated with the app is launched.
CURRENT_WORKING_DIR = ""
STATIC_DIR = ""
PLOTS_DIR = ""
DATASETS_DIR = ""
GRAPHICS_DIR = ""
EXPORT_PATH_AND_FILENAME = ""

# ======================================================================
# Variables relating to the web app.
# ======================================================================

plots_to_display_list = []
models_data_source = "built_in_pers_day_df"
dataset_csv_for_download_url = None
uploaded_wfs_behaviors_and_records_csv = None

# ======================================================================
# Elements needed for calculating filename prefixes for exports.
# ======================================================================

unique_file_prefix_code_for_simulation_run = None
unique_file_suffix_code_for_simulation_run = None

# ======================================================================
# Variables relating to simulation iteration date and time.
# ======================================================================

# This is the time at which processing of the simulation begins on the
# user's computer; it's used for tracking elapsed processing time.
sim_processing_start_datetime = None

# Configurable settings.
USER_CONFIGURABLE_MODEL_SETTING_A = "Efficacy"
USER_CONFIGURABLE_MODEL_SETTING_B = "30_d"
USER_CONFIGURABLE_MODEL_SETTING_C = 101
USER_CONFIGURABLE_MODEL_SETTING_D = "Small"

# ======================================================================
# Constants/variables relating to visualizations.
# ======================================================================
PLOT_FIGURE_DPI = 500
PLOT_SAVEFIG_DPI = 500
PLOT_FIGSIZE = (6.5, 3)
PLOT_XY_LABEL_FONTSIZE = 7
PLOT_XY_LABEL_PAD = 4
PLOT_LINE_DATA_WIDTH = 1.85
PLOT_XY_TICKS_FONTSIZE = 7
PLOT_TITLE_FONTSIZE = 8

PLOT_COLOR_LIGHT_PINK = "#FDECE4"
PLOT_COLOR_LAVENDER = "#9ea3ff"
PLOT_COLOR_LAVENDER_DARK = "#7C7FF0"
PLOT_COLOR_MAGENTA = "#ff00ff"
PLOT_COLOR_MAGENTA_LIGHT = "#FFDDFF"
PLOT_COLOR_GREEN = "#00FA95"
PLOT_COLOR_GREEN_LIGHT = "#D1FFEC"
PLOT_COLOR_GOLD = "#ffa74d"
PLOT_COLOR_DARK_PLUM = '#34343C' # darkest plum
PLOT_COLOR_SALMON = "#fc8293" # salmon
PLOT_COLOR_SALMON_DARK = "#FA3C57" # salmon
PLOT_COLOR_CYAN = "#5cffe5" # cyan
PLOT_COLOR_MEDIUM_GRAY = '#404040' # dark gray
PLOT_COLOR_DARKER_GRAY = "#2A2A2A" # darker gray
PLOT_COLOR_DARKEST_GRAY = "#191919" # almost black
PLOT_COLOR_BLACK = "black" # black

y_max_for_plotting = 0.0

# # ======================================================================
# Constants and variables relating to machine learning.
# ======================================================================

# This transformed version of the DF has had features and targets 
# engineered, for use in machine-learning-based analyses. Each row 
# represents a single day of activity for a single person.
pers_day_df = None

potential_target_columns = []

# A DataFrame containing all potential features (i.e., columns with X
# values) and potential targets (i.e., columns with y values)
# for that portion of the dataset assigned for use in *training* a model.
Xy_train_df = None

# A DataFrame containing all potential features (i.e., columns with X
# values) and potential targets (i.e., columns with y values)
# for that portion of the dataset assigned for use in *validating* a 
# model.
Xy_valid_df = None

# A DataFrame containing all potential features (i.e., columns with X
# values) and potential targets (i.e., columns with y values)
# for that portion of the dataset assigned for use in *testing* a model.
Xy_test_df = None

# A DataFrame containing only those features (i.e., columns with X values)
# that have been selected from among all features for use with a model,
# within that portion of the dataset assigned for use in *training* the 
# model.
X_train_slctd_df = None
X_train_slctd_df_preprocessed = None

# A DataFrame containing only those targets (i.e., columns with y values)
# that have been selected from among all targets for use with a model,
# within that portion of the dataset assigned for use in *training* the 
# model.
y_train_slctd_df = None

# A DataFrame containing only those features (i.e., columns with X values)
# that have been selected from among all features for use with a model,
# within that portion of the dataset assigned for use in *validating* the 
# model.
X_valid_slctd_df = None
X_valid_slctd_df_preprocessed = None
X_valid_slctd_df_preprocessed_with_y_preds_base = None

# A DataFrame containing only those targets (i.e., columns with y values)
# that have been selected from among all targets for use with a model,
# within that portion of the dataset assigned for use in *validating* 
# the model.
y_valid_slctd_df = None

# A DataFrame containing only those features (i.e., columns with X values)
# that have been selected from among all features for use with a model,
# within that portion of the dataset assigned for use in *testing* the 
# model.
X_test_slctd_df = None

# A DataFrame containing only those targets (i.e., columns with y values)
# that have been selected from among all targets for use with a model,
# within that portion of the dataset assigned for use in *testing* the 
# model.
y_test_slctd_df = None

# The features that should be returned as predictions by Naive Persistence
# and Naive Mean models.
feature_to_return_naive_persistence = None
feature_to_return_naive_mean = None

# The feature that should be used when calculating the likely actual 
# range for target values (e.g., a historical standard deviation figure).
feature_for_calculating_target_range = None

# For a model that generates a predicted value for a target along with a likely
# floor and ceiling, these are the lists of floor, target, and ceiling values
# generated by the model that is currently being utilized.
#
# For a model that only generates a predicted target value (without a range),
# only y_preds_base_target_curr will have values generated by the model.
y_preds_floor_curr = None
y_preds_base_target_curr = None
y_preds_ceiling_curr = None

# Model metrics
mae = None
mae_base_target_model = None
mse = None
patgtc = None
amorpdac = None
amirpdbc = None
oce = None
patltf = None
amorpdbf = None
amirpdaf = None
ofe = None
mprs = None
irp = None
pator = None
msadre = None
orp = None

lowest_safe_ceiling_loss_func_id = None
highest_safe_floor_loss_func_id = None

models_metrics_df = None

model_id_curr = None

SD_MULTIPLIERS_TO_USE = [
    0.674490,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.644854,
    ]

# ----------------------------------------------------------------------
# Feature columns to use.
# ----------------------------------------------------------------------

feature_target_arguments_for_modelling_dict = {}
# Predict mean Eff during the next 7 days.
feature_target_arguments_for_modelling_dict[
    "feature_target_args_nxt_7_d_eff"] = {
        "features_list_non_OHE_u": [
            "d0_eff_rec_val",
            "career_eff_rec_mean",
            "career_eff_rec_sd",
            "career_absences_per_cal_d",
            "d0_weekday_num",
            "d0_day_in_series",
            "prev_7_d_ideas_rec_num",
            "prev_7_d_eff_rec_mean",
            "prev_30_d_eff_rec_mean",
            "prev_30_d_eff_rec_sd",
            ],
        "features_list_OHE_u":
            ["sub_sex"],
        "targets_list_u":
            ["nxt_7_d_eff_rec_mean"],
        "feature_to_use_for_naive_persistence_model_u":
            "prev_7_d_eff_rec_mean",
        "feature_to_use_for_naive_mean_model_u":
            "career_eff_rec_mean",
        "feature_for_calculating_target_range_u":
            "career_eff_rec_sd",
        }

# Predict mean Eff during the next 30 days.
feature_target_arguments_for_modelling_dict[
    "feature_target_args_nxt_30_d_eff"] = {
        "features_list_non_OHE_u": [
            "d0_eff_rec_val",
            "career_eff_rec_mean",
            "career_eff_rec_sd",
            "career_absences_per_cal_d",
            "d0_weekday_num",
            "d0_day_in_series",
            "prev_30_d_ideas_rec_num",
            "prev_30_d_eff_rec_mean",
            "prev_30_d_eff_rec_sd",
            ],
        "features_list_OHE_u":
            ["sub_sex"],
        "targets_list_u":
            ["nxt_30_d_eff_rec_mean"],
        "feature_to_use_for_naive_persistence_model_u":
            "prev_30_d_eff_rec_mean",
        "feature_to_use_for_naive_mean_model_u":
            "career_eff_rec_mean",
        "feature_for_calculating_target_range_u":
            "career_eff_rec_sd",
        }

# Predict number of Teamworks during the next 7 days.
feature_target_arguments_for_modelling_dict[
    "feature_target_args_nxt_7_d_teamworks"] = {
        "features_list_non_OHE_u": [
            "d0_eff_rec_val",
            "career_eff_rec_mean",
            "career_eff_rec_sd",
            "career_absences_per_cal_d",
            "d0_weekday_num",
            "d0_day_in_series",
            "prev_7_d_teamworks_rec_num",
            ],
        "features_list_OHE_u":
            ["sub_sex"],
        "targets_list_u":
            ["nxt_7_d_teamworks_rec_num"],
        "feature_to_use_for_naive_persistence_model_u":
            "prev_7_d_teamworks_rec_num",
        "feature_to_use_for_naive_mean_model_u":
            "prev_7_d_teawmorks_rec_num",
        "feature_for_calculating_target_range_u":
            "career_eff_rec_sd",
        }

# Predict number of Teamworks during the next 30 days.
feature_target_arguments_for_modelling_dict[
    "feature_target_args_nxt_30_d_teamworks"] = {
        "features_list_non_OHE_u": [
            "d0_eff_rec_val",
            "career_eff_rec_mean",
            "career_eff_rec_sd",
            "career_absences_per_cal_d",
            "d0_weekday_num",
            "d0_day_in_series",
            "prev_30_d_teamworks_rec_num",
            ],
        "features_list_OHE_u":
            ["sub_sex"],
        "targets_list_u":
            ["nxt_30_d_teamworks_rec_num"],
        "feature_to_use_for_naive_persistence_model_u":
            "prev_30_d_teamworks_rec_num",
        "feature_to_use_for_naive_mean_model_u":
            "prev_30_d_teamworks_rec_num",
        "feature_for_calculating_target_range_u":
            "career_eff_rec_sd",
        }

# ======================================================================
# Dictionary with key info for all models.
# ======================================================================

models_info_dict = {}

# Some models *aren't* explicitly specified here:
#
# - Ceiling Models using SD to make their prediction: their 
#   specifications will be automatically generated when the package is 
#   run.
#
# - Floor Models using SD to make their prediction: their 
#   specifications will be automatically generated when the package is 
#   run.

# ----------------------------------------------------------------------
# Base Target Model.
# ----------------------------------------------------------------------

# The Base Target Model.
models_info_dict["BT"] = {
    "model_id": "BT",
    "model_type": "Base Target",
    "model_parameters_description": "RF; n_estimators 300",
    "model_title_for_plot_display": "Base Target Model",
    "model_name_for_plot_filename": "BT",
    }

# ----------------------------------------------------------------------
# Ceiling Models.
# ----------------------------------------------------------------------

# Ceiling Model using MAE to make its prediction.
models_info_dict["clng_MAE"] = {
    "model_id": "clng_MAE",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "predicted target value + Base Target Model MAE",
    "model_title_for_plot_display": \
        "predicted target value + Base Target Model MAE",
    "model_name_for_plot_filename": "clng_MAE",
    }

# Ceiling Models with custom loss functions.
models_info_dict["clng_cstm_A_1"] = {
    "model_id": "clng_cstm_A_1",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.5×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.5×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_1",
    }
models_info_dict["clng_cstm_A_2"] = {
    "model_id": "clng_cstm_A_2",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.55×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.55×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_2",
    }
models_info_dict["clng_cstm_A_3"] = {
    "model_id": "clng_cstm_A_3",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.6×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.6×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_3",
    }
models_info_dict["clng_cstm_A_4"] = {
    "model_id": "clng_cstm_A_4",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.65×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.65×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_4",
    }
models_info_dict["clng_cstm_A_5"] = {
    "model_id": "clng_cstm_A_5",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.7×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.7×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_5",
    }
models_info_dict["clng_cstm_A_6"] = {
    "model_id": "clng_cstm_A_6",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.75×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.75×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_6",
    }
models_info_dict["clng_cstm_A_7"] = {
    "model_id": "clng_cstm_A_7",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.8×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.8×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_7",
    }
models_info_dict["clng_cstm_A_8"] = {
    "model_id": "clng_cstm_A_8",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.85×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.85×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_8",
    }
models_info_dict["clng_cstm_A_9"] = {
    "model_id": "clng_cstm_A_9",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 0.9×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 0.9×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_9",
    }
models_info_dict["clng_cstm_A_10"] = {
    "model_id": "clng_cstm_A_10",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 1.0×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 1.0×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_10",
    }
models_info_dict["clng_cstm_A_11"] = {
    "model_id": "clng_cstm_A_11",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff + 1.1×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff + 1.1×(mean diff)",
    "model_name_for_plot_filename": "clng_cstm_A_11",
    }
models_info_dict["clng_cstm_B_1"] = {
    "model_id": "clng_cstm_B_1",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = mean ops_sqr_diff",
    "model_title_for_plot_display": "loss = mean ops_sqr_diff",
    "model_name_for_plot_filename": "clng_cstm_B_1",
    }
models_info_dict["clng_cstm_C_1"] = {
    "model_id": "clng_cstm_C_1",
    "model_type": "Ceiling",
    "model_parameters_description": \
        "loss = sqr diff prop | 50×(sqr diff prop)",
    "model_title_for_plot_display": \
        "loss = sqr diff prop | 50×(sqr diff prop)",
    "model_name_for_plot_filename": "clng_cstm_C_1",
    }
models_info_dict["clng_cstm_D_1"] = {
    "model_id": "clng_cstm_D_1",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 10×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 10×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_1",
    }
models_info_dict["clng_cstm_D_2"] = {
    "model_id": "clng_cstm_D_2",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 30×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 30×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_2",
    }
models_info_dict["clng_cstm_D_3"] = {
    "model_id": "clng_cstm_D_3",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 50×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 50×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_3",
    }
models_info_dict["clng_cstm_D_4"] = {
    "model_id": "clng_cstm_D_4",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 100×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 100×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_4",
    }
models_info_dict["clng_cstm_D_5"] = {
    "model_id": "clng_cstm_D_5",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = abs diff | 100×(sqr diff)",
    "model_title_for_plot_display": "loss = abs diff | 100×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_5",
    }
models_info_dict["clng_cstm_D_6"] = {
    "model_id": "clng_cstm_D_6",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 200×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 200×(sqr diff)",
    "model_name_for_plot_filename": "clng_cstm_D_6",
    }
models_info_dict["clng_cstm_E_1"] = {
    "model_id": "clng_cstm_E_1",
    "model_type": "Ceiling",
    "model_parameters_description": "loss = sqr diff | 10×(diff^8)",
    "model_title_for_plot_display": "loss = sqr diff | 10×(diff^8)",
    "model_name_for_plot_filename": "clng_cstm_E_1",
    }

# ----------------------------------------------------------------------
# Floor Models.
# ----------------------------------------------------------------------

# Floor Model using MAE to make its prediction.
models_info_dict["flr_MAE"] = {
    "model_id": "flr_MAE",
    "model_type": "Floor",
    "model_parameters_description": \
        "predicted target value - Base Target Model MAE",
    "model_title_for_plot_display": \
        "predicted target value - Base Target Model MAE",
    "model_name_for_plot_filename": "flr_MAE",
    }

# Floor Models with custom loss functions.
models_info_dict["flr_cstm_A_1"] = {
    "model_id": "flr_cstm_A_1",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.5×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.5×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_1",
    }
models_info_dict["flr_cstm_A_2"] = {
    "model_id": "flr_cstm_A_2",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.55×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.55×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_2",
    }
models_info_dict["flr_cstm_A_3"] = {
    "model_id": "flr_cstm_A_3",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.6×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.6×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_3",
    }
models_info_dict["flr_cstm_A_4"] = {
    "model_id": "flr_cstm_A_4",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.65×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.65×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_4",
    }
models_info_dict["flr_cstm_A_5"] = {
    "model_id": "flr_cstm_A_5",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.7×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.7×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_5",
    }
models_info_dict["flr_cstm_A_6"] = {
    "model_id": "flr_cstm_A_6",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.75×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.75×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_6",
    }
models_info_dict["flr_cstm_A_7"] = {
    "model_id": "flr_cstm_A_7",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.8×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.8×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_7",
    }
models_info_dict["flr_cstm_A_8"] = {
    "model_id": "flr_cstm_A_8",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.85×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.85×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_8",
    }
models_info_dict["flr_cstm_A_9"] = {
    "model_id": "flr_cstm_A_9",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 0.9×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 0.9×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_9",
    }
models_info_dict["flr_cstm_A_10"] = {
    "model_id": "flr_cstm_A_10",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 1.0×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 1.0×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_10",
    }
models_info_dict["flr_cstm_A_11"] = {
    "model_id": "flr_cstm_A_11",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = mean ops_sqr_diff - 1.1×(mean diff)",
    "model_title_for_plot_display": \
        "loss = mean ops_sqr_diff - 1.1×(mean diff)",
    "model_name_for_plot_filename": "flr_cstm_A_11",
    }
models_info_dict["flr_cstm_B_1"] = {
    "model_id": "flr_cstm_B_1",
    "model_type": "Floor",
    "model_parameters_description": "loss = mean ops_sqr_diff",
    "model_title_for_plot_display": "loss = mean ops_sqr_diff",
    "model_name_for_plot_filename": "flr_cstm_B_1",
    }
models_info_dict["flr_cstm_C_1"] = {
    "model_id": "flr_cstm_C_1",
    "model_type": "Floor",
    "model_parameters_description": \
        "loss = sqr diff prop | 50×(sqr diff prop)",
    "model_title_for_plot_display": \
        "loss = sqr diff prop | 50×(sqr diff prop)",
    "model_name_for_plot_filename": "flr_cstm_C_1",
    }
models_info_dict["flr_cstm_D_1"] = {
    "model_id": "flr_cstm_D_1",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 10×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 10×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_1",
    }
models_info_dict["flr_cstm_D_2"] = {
    "model_id": "flr_cstm_D_2",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 30×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 30×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_2",
    }
models_info_dict["flr_cstm_D_3"] = {
    "model_id": "flr_cstm_D_3",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 50×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 50×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_3",
    }
models_info_dict["flr_cstm_D_4"] = {
    "model_id": "flr_cstm_D_4",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 100×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 100×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_4",
    }
models_info_dict["flr_cstm_D_5"] = {
    "model_id": "flr_cstm_D_5",
    "model_type": "Floor",
    "model_parameters_description": "loss = abs diff | 100×(sqr diff)",
    "model_title_for_plot_display": "loss = abs diff | 100×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_5",
    }
models_info_dict["flr_cstm_D_6"] = {
    "model_id": "flr_cstm_D_6",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 200×(sqr diff)",
    "model_title_for_plot_display": "loss = sqr diff | 200×(sqr diff)",
    "model_name_for_plot_filename": "flr_cstm_D_6",
    }
models_info_dict["flr_cstm_E_1"] = {
    "model_id": "flr_cstm_E_1",
    "model_type": "Floor",
    "model_parameters_description": "loss = sqr diff | 10×(diff^8)",
    "model_title_for_plot_display": "loss = sqr diff | 10×(diff^8)",
    "model_name_for_plot_filename": "flr_cstm_E_1",
    }


# ██████████████████████████████████████████████████████████████████████