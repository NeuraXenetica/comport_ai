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
This module includes functions used in reading data from file and 
saving data to file.
"""

import os
import datetime

import pandas as pd

# Import other modules from this package.
import config as cfg
import cai_ai as ai
import cai_utilities as utils


def specify_directory_structure():
    """
    Specifies the directory structure for file reading and writing.
    """

    cfg.CURRENT_WORKING_DIR = os.getcwd()
    cfg.STATIC_DIR = os.path.abspath(
        os.path.join(cfg.CURRENT_WORKING_DIR, 'static'))
    cfg.PLOTS_DIR = os.path.abspath(
        os.path.join(cfg.STATIC_DIR, 'plots'))
    cfg.GRAPHICS_DIR = os.path.abspath(
        os.path.join(cfg.STATIC_DIR, 'graphics'))
    cfg.DATASETS_DIR = os.path.abspath(
        os.path.join(cfg.CURRENT_WORKING_DIR, 'datasets'))


def generate_unique_file_prefix_code_for_simulation_run():
    """
    Generates a unique code for this run of the simulation, which can 
    be used as a prefix for the files to be saved that are associated 
    with this run.
    """

    # Get current date and time.
    datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    cfg.unique_file_prefix_code_for_simulation_run = \
        "[" + datetime_str +"] "

    # This variant can be used as a suffix instead of a prefix.
    cfg.unique_file_suffix_code_for_simulation_run = \
        "[" + datetime_str +"] "


def save_df_to_pickle_file(input_df_u, filename_u):
    """
    Saves a DataFrame to disk as a pickle file.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame to be saved
    filename_u
        The desired filename (without prefix code or .xlsx ending)
    """

    full_filename = filename_u + ".pickle"
    filename_and_path = os.path.join(
        cfg.DATASETS_DIR, "user_generated", full_filename)
    input_df_u.to_pickle(filename_and_path)


def save_df_to_xlsx_file(input_df_u, filename_u):
    """
    Saves a DataFrame to disk as an XLSX file.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame to be saved
    filename_u
        The desired filename (without prefix code or .xlsx ending)
    """

    full_filename = cfg.unique_file_prefix_code_for_simulation_run \
        + filename_u + ".xlsx"
    filename_and_path = os.path.join(
        cfg.DATASETS_DIR, "user_generated", full_filename)
    input_df_u.to_excel(filename_and_path)


def save_df_to_csv_file(input_df_u, filename_u):
    """
    Saves a DataFrame to disk as a CSV file.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame to be saved
    filename_u
        The desired filename (without prefix code or .csv ending)
    """

    filename_and_path = os.path.join(
        cfg.DATASETS_DIR, "user_generated", filename_u + ".csv")
    input_df_u.to_csv(
        filename_and_path,
        encoding="utf-8-sig",
        index=False)


def save_pers_day_df_to_pickle():
    """
    Saves the cfg.pers_day_df DataFrame to a pickle file.
    """

    save_df_to_pickle_file(
        cfg.pers_day_df,
        "pers_day_df",
    )

    save_df_to_csv_file(
        cfg.pers_day_df,
        "pers_day_df_to_csv",
    )


def load_pers_day_df_from_pickle(
    full_name_of_pers_day_df_file_to_load_u,
    ):
    """
    Loads a pickle file to cfg.pers_day_df.

    PARAMETERS
    ----------
    full_name_of_pers_day_df_file_to_load_u
        Name of the pers_day_df file to load
    """

    filename_and_path = os.path.join(
        cfg.DATASETS_DIR, "pregenerated", full_name_of_pers_day_df_file_to_load_u
        )
    cfg.pers_day_df = pd.read_pickle(filename_and_path)


def run_one_time_steps_to_generate_pers_day_df_from_behavs_act_df():
    """
    Runs one-time steps needed to generate the cfg.pers_day_df
    DataFrame from a WFS file of workers' behaviors and records.
    """

    load_wfs_behaviors_records_df_csv_as_cfg_behavs_act_df()

    utils.begin_tracking_elapsed_processing_time()

    ai.create_columns_in_behavs_act_day_df()
    ai.one_hot_encode_behavs_act_df_columns()
    ai.create_pers_day_df_from_behavs_act_day_df()
    ai.create_columns_in_pers_day_df()
    ai.engineer_pers_day_df_features_and_targets()

    print(
        "Elapsed time to saving of pers_day_df pickle file: ",
        utils.return_elapsed_processing_time()
        )


def load_wfs_behaviors_records_df_csv_as_cfg_behavs_act_df():
    """
    Loads a WorkforceSim CSV file of workers' behaviors and records.
    """

    filename_and_path = os.path.join(
        cfg.DATASETS_DIR, "user_generated", "uploaded_file.csv"
        )

    cfg.behavs_act_df = pd.read_csv(
        filename_and_path,
        )

    for i in range(len(cfg.behavs_act_df)):
        cfg.behavs_act_df["event_date"].values[i] = \
            (pd.Timestamp(cfg.behavs_act_df["event_date"].values[i])).to_pydatetime()


# ██████████████████████████████████████████████████████████████████████