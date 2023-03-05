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
# ║   Developed by Matthew E. Gladden • ©2021-23 NeuraXenetica LLC     ║
# ║   This software is made available for use under                    ║
# ║   GNU General Public License Version 3                             ║
# ║   (please see https://www.gnu.org/licenses/gpl-3.0.html).          ║
# ╚════════════════════════════════════════════════════════════════════╝

"""
This module oversees the process of loading or preparing a dataset,
training and validating models, and saving the results.
"""

import random
import tensorflow as tf
import numpy as np

# Import other modules from this package.
import config as cfg
import io_file_manager as iofm
import cai_ai as ai


# ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
# ● Prepare the basic cfg.pers_day_df DataFrame
# ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

# ---------------------------------------------------------------------
# Option 1: Create a fresh cfg.pers_day_df DataFrame.
# ---------------------------------------------------------------------
def create_fresh_pers_day_df():
    """
    Creates a fresh cfg.pers_day_df DataFrame of entry data
    in an ML-ready format by processing a cfg.behavs_act_df variable
    that contains data (by virtue of having just run the simulation or
    having loaded a saved dataset from file). The newly created content
    of cfg.pers_day_df is also saved to disk for future use).
    """

    iofm.specify_directory_structure()
    iofm.run_one_time_steps_to_generate_pers_day_df_from_behavs_act_df()


# ---------------------------------------------------------------------
# Option 2: Load an existing cfg.pers_day_df from disk.
# ---------------------------------------------------------------------
def load_existing_pers_day_df_from_disk():
    """
    Loads an existing cfg.pers_day_df from disk.
    """

    iofm.specify_directory_structure()

    #if cfg.USER_CONFIGURABLE_MODEL_SETTING_D == "Large":
    #    iofm.load_pers_day_df_from_pickle("pers_day_df_large.pickle")
    if cfg.USER_CONFIGURABLE_MODEL_SETTING_D == "Medium":
        iofm.load_pers_day_df_from_pickle("pers_day_df_medium.pickle")
    elif cfg.USER_CONFIGURABLE_MODEL_SETTING_D == "Small":
        iofm.load_pers_day_df_from_pickle("pers_day_df_small.pickle")


# ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
# ● Train and validate the models and visualize the results
# ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

def create_train_valid_test_dfs_and_extract_features_and_targets():
    """
    Creates a split of training, validation, and test DFs and then
    extracts desired features and targets.
    """

    ai.create_train_valid_test_split_from_pers_day_df(
        0.333,
        0.333,
        )

    if cfg.USER_CONFIGURABLE_MODEL_SETTING_A == "Efficacy":
        if cfg.USER_CONFIGURABLE_MODEL_SETTING_B == "7_d":
            ai.select_features_and_targets_in_pers_day_df(
                **cfg.feature_target_arguments_for_modelling_dict[
                    "feature_target_args_nxt_7_d_eff"
                    ]
                )
        elif cfg.USER_CONFIGURABLE_MODEL_SETTING_B == "30_d":
            ai.select_features_and_targets_in_pers_day_df(
                **cfg.feature_target_arguments_for_modelling_dict[
                    "feature_target_args_nxt_30_d_eff"
                    ]
                )
    elif cfg.USER_CONFIGURABLE_MODEL_SETTING_A == "Teamworks":
        if cfg.USER_CONFIGURABLE_MODEL_SETTING_B == "7_d":
            ai.select_features_and_targets_in_pers_day_df(
                **cfg.feature_target_arguments_for_modelling_dict[
                    "feature_target_args_nxt_7_d_teamworks"
                    ]
                )
        elif cfg.USER_CONFIGURABLE_MODEL_SETTING_B == "30_d":
            ai.select_features_and_targets_in_pers_day_df(
                **cfg.feature_target_arguments_for_modelling_dict[
                    "feature_target_args_nxt_30_d_teamworks"
                    ]
                )


def train_and_validate_models_and_generate_plots():
    """
    Trains and validates the models and then visualizes the results.
    """

    iofm.generate_unique_file_prefix_code_for_simulation_run()

    random.seed(cfg.USER_CONFIGURABLE_MODEL_SETTING_C)
    np.random.seed(cfg.USER_CONFIGURABLE_MODEL_SETTING_C)
    tf.random.set_seed(cfg.USER_CONFIGURABLE_MODEL_SETTING_C)

    ai.preprocess_data_for_use_with_all_models()
    ai.create_df_for_tracking_model_metrics()
    ai.train_and_validate_base_target_model()
    ai.validate_mae_ceiling_model()
    ai.validate_mae_floor_model()

    for sd_multiplier in cfg.SD_MULTIPLIERS_TO_USE:
        ai.validate_n_sd_ceiling_model(sd_multiplier)
        ai.validate_n_sd_floor_model(sd_multiplier)

    for highest_safe_floor_loss_func_id in [
        "flr_cstm_A_1",
        "flr_cstm_A_2",
        "flr_cstm_A_3",
        "flr_cstm_A_4",
        "flr_cstm_A_5",
        "flr_cstm_A_6",
        "flr_cstm_A_7",
        "flr_cstm_A_8",
        "flr_cstm_A_9",
        "flr_cstm_A_10",
        "flr_cstm_A_11",
        "flr_cstm_B_1",
        "flr_cstm_C_1",
        "flr_cstm_D_1",
        "flr_cstm_D_2",
        "flr_cstm_D_3",
        "flr_cstm_D_4",
        "flr_cstm_D_5",
        "flr_cstm_D_6",
        "flr_cstm_E_1",
        ]:
        ai.train_and_validate_nn_floor_model(highest_safe_floor_loss_func_id)


    for lowest_safe_ceiling_loss_func_id in [
        "clng_cstm_A_1",
        "clng_cstm_A_2",
        "clng_cstm_A_3",
        "clng_cstm_A_4",
        "clng_cstm_A_5",
        "clng_cstm_A_6",
        "clng_cstm_A_7",
        "clng_cstm_A_8",
        "clng_cstm_A_9",
        "clng_cstm_A_10",
        "clng_cstm_A_11",
        "clng_cstm_B_1",
        "clng_cstm_C_1",
        "clng_cstm_D_1",
        "clng_cstm_D_2",
        "clng_cstm_D_3",
        "clng_cstm_D_4",
        "clng_cstm_D_5",
        "clng_cstm_D_6",
        "clng_cstm_E_1",
        ]:
        ai.train_and_validate_nn_ceiling_model(lowest_safe_ceiling_loss_func_id)

    # Find the max y-value needed for the Joint Range plots'
    # vertical axis.
    cfg.y_max_for_plotting = 0.0
    for key in cfg.models_info_dict:
        if "y_max_candidate_for_joint_range_plots" in cfg.models_info_dict[key]:
            if (cfg.models_info_dict[key]["y_max_candidate_for_joint_range_plots"] \
                    > cfg.y_max_for_plotting):
                cfg.y_max_for_plotting \
                    = cfg.models_info_dict[key]["y_max_candidate_for_joint_range_plots"]
    print("cfg.y_max_for_plotting: ", cfg.y_max_for_plotting)

    ai.create_joint_range_models_from_floor_ceiling_models()

    filename_without_prefix = \
        "models_metrics_df_" \
        + str(cfg.USER_CONFIGURABLE_MODEL_SETTING_A) \
        + "_" \
        + str(cfg.USER_CONFIGURABLE_MODEL_SETTING_B) \
        + "_" \
        + str(cfg.USER_CONFIGURABLE_MODEL_SETTING_C) \
        + "_" \
        + str(cfg.USER_CONFIGURABLE_MODEL_SETTING_D) \

    iofm.save_df_to_xlsx_file(
        cfg.models_metrics_df,
        filename_without_prefix
        )

    iofm.save_df_to_csv_file(
        cfg.models_metrics_df,
        filename_without_prefix
        )

    cfg.dataset_csv_for_download_url = \
        "/datasets/user_generated/" + filename_without_prefix + ".csv"


# ██████████████████████████████████████████████████████████████████████