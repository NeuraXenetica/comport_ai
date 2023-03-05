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
This module handles the training and validation of the Base Target
Model, individual Ceiling and Floor Models, and the Joint Range Models
that combine a Ceiling Model with a Floor Model.
"""

from datetime import timedelta
import random
import os
import statistics
import pickle

import pandas as pd
import numpy as np
from tensorflow.python.ops import math_ops
import keras
from keras import backend as K
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from tabulate import tabulate

# Import other modules from this package.
import config as cfg
import cai_visualizer as vis


def create_columns_in_behavs_act_day_df():
    """
    Creates in cfg.pers_day_df a range of columns (including some OHE
    columns) that are necessary or useful for applying machine-learning
    algorithms to make predictions. In itself, this function doesn't yet
    populate the columns' values.
    """

    print("Beginning create_columns_in_behavs_act_day_df().")

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # █ Prepare DataFrames and columns.
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

    # Keep selected columns from cfg.behavs_act_df.
    cfg.behavs_act_df = cfg.behavs_act_df[[
        "sub_ID",
        "sub_age",
        "sub_sex",
        "sub_shift",
        "sub_team",
        "sub_role",
        "sub_colls_same_sex_prtn",
        "sub_workstyle_h",
        "sup_ID",
        "sup_age",
        "sup_sub_age_diff",
        "sup_sex",
        "sup_role",
        "event_date",
        "event_week_in_series",
        "event_day_in_series",
        "event_weekday_num",
        "actual_efficacy_h",
        "record_comptype",
        "recorded_efficacy",
        ]]

    # Rename selected columns.
    cfg.behavs_act_df = cfg.behavs_act_df.rename(
        columns={
            "sub_workstyle_h": "sub_workstyle",
            "event_date": "d0_date",
            "event_week_in_series": "d0_week_in_series",
            "event_day_in_series": "d0_day_in_series",
            "event_weekday_num": "d0_weekday_num",
            "actual_efficacy_h": "d0_eff_act_val",
            "record_comptype": "d0_rec_comptype",
            "recorded_efficacy": "d0_eff_rec_val",
        })


def one_hot_encode_behavs_act_df_columns():
    """
    One-hot encodes selected columns relating to particular behaviors
    and records in cfg.behavs_act_df.
    """

    print("Beginning one_hot_encode_behavs_act_df_columns().")

    # Note that in order to avoid the Dummy Variable Trap, 
    # "drop_first=True" should be employed, as appropriate.
    d0_rec_comptype_df = \
        pd.get_dummies(cfg.behavs_act_df["d0_rec_comptype"].astype(
            pd.CategoricalDtype(categories=[
                "Absence",
                "Disruption",
                "Efficacy",
                "Feat",
                "Idea",
                "Lapse",
                "Onboarding",
                "Presence",
                "Resignation",
                "Sabotage",
                "Sacrifice",
                "Slip",
                "Teamwork",
                "Termination",
                ])),
            prefix = "rcomp",
            prefix_sep = "_",
        )

    cfg.behavs_act_df = pd.merge(
        left = cfg.behavs_act_df,
        right = d0_rec_comptype_df,
        left_index=True,
        right_index=True,
        )


def create_pers_day_df_from_behavs_act_day_df():
    """
    Creates a DataFrame of person-day observations from a DataFrame
    of individual behaviors and records.
    """

    # ------------------------------------------------------------------
    # Performing grouping to turn cfg.behavs_act_df into 
    # cfg.pers_day_df.
    # ------------------------------------------------------------------
    cfg.pers_day_df = cfg.behavs_act_df.copy().reset_index(drop=True)

    # Aggregate the rows by subject and day.
    aggregation_method = {
        "sub_age": "last",
        "sub_sex": "last",
        "sub_shift": "last",
        "sub_team": "last",
        "sub_role": "last",
        "sub_colls_same_sex_prtn": "last",
        "sup_ID": "last",
        "sup_age": "last",
        "sup_sub_age_diff": "last",
        "sup_sex": "last",
        "sup_role": "last",
        "d0_week_in_series": "last",
        "d0_day_in_series": "last",
        "d0_weekday_num": "last",
        "d0_rec_comptype": "last",
        "d0_eff_rec_val": "mean",
        "rcomp_Absence": "sum",
        "rcomp_Disruption": "sum",
        "rcomp_Efficacy": "sum",
        "rcomp_Feat": "sum",
        "rcomp_Idea": "sum",
        "rcomp_Lapse": "sum",
        "rcomp_Onboarding": "sum",
        "rcomp_Presence": "sum",
        "rcomp_Resignation": "sum",
        "rcomp_Sabotage": "sum",
        "rcomp_Sacrifice": "sum",
        "rcomp_Slip": "sum",
        "rcomp_Teamwork": "sum",
        "rcomp_Termination": "sum",
        }

    cfg.pers_day_df = \
        cfg.pers_day_df.groupby(by=["sub_ID", "d0_date"]).agg(
            aggregation_method
            ).reset_index(drop=False)

    for col in [
        "sub_sex",
        "sub_shift",
        "sub_team",
        "sub_role",
        "sup_sex",
        "sup_role",
        ]:
        # One-hot encode the given column.
        cfg.pers_day_df = pd.merge(
            left=cfg.pers_day_df,
            right = pd.get_dummies(
                cfg.pers_day_df[col], prefix=col, drop_first=True
                ),
            left_index=True,
            right_index=True,
            )
        # Delete the original column (which contains non-numeric data).
        cfg.pers_day_df = cfg.pers_day_df.drop(columns=col)


def create_columns_in_pers_day_df():
    """
    Creates columns in cfg.pers_day_df for (potential) features and
    targets.
    """

    # ------------------------------------------------------------------
    # (Potential) feature columns.
    # ------------------------------------------------------------------

    cfg.pers_day_df[[
        "d0_day_of_month_num",
        "d0_month_of_year_num",

        "dm1_eff_rec_val",
        "dm2_eff_rec_val",
        "dm3_eff_rec_val",
        "dm4_eff_rec_val",
        "prev_7_d_eff_rec_mean",
        "prev_7_d_eff_rec_sd",
        "prev_30_d_eff_rec_mean",
        "prev_30_d_eff_rec_sd",
        "career_eff_rec_mean",
        "career_eff_rec_sd",

        "dm1_presence_yn",
        "prev_7_d_presences_num",
        "prev_30_d_presences_num",
        "career_presences_per_cal_d",
        "dm1_absence_yn",
        "prev_7_d_absences_num",
        "prev_30_d_absences_num",
        "career_absences_per_cal_d",

        "dm1_ideas_rec_num",
        "dm2_ideas_rec_num",
        "dm3_ideas_rec_num",
        "prev_7_d_ideas_rec_num",
        "prev_30_d_ideas_rec_num",
        "career_ideas_rec_per_cal_d",
        "career_ideas_rec_per_d_pres",

        "dm1_lapses_rec_num",
        "dm2_lapses_rec_num",
        "dm3_lapses_rec_num",
        "prev_7_d_lapses_rec_num",
        "prev_30_d_lapses_rec_num",
        "career_lapses_rec_per_cal_d",
        "career_lapses_rec_per_d_pres",

        "dm1_feats_rec_num",
        "dm2_feats_rec_num",
        "dm3_feats_rec_num",
        "prev_7_d_feats_rec_num",
        "prev_30_d_feats_rec_num",
        "career_feats_rec_per_cal_d",
        "career_feats_rec_per_d_pres",

        "dm1_slips_rec_num",
        "dm2_slips_rec_num",
        "dm3_slips_rec_num",
        "prev_7_d_slips_rec_num",
        "prev_30_d_slips_rec_num",
        "career_slips_rec_per_cal_d",
        "career_slips_rec_per_d_pres",

        "dm1_teamworks_rec_num",
        "dm2_teamworks_rec_num",
        "dm3_teamworks_rec_num",
        "prev_7_d_teamworks_rec_num",
        "prev_30_d_teamworks_rec_num",
        "career_teamworks_rec_per_cal_d",
        "career_teamworks_rec_per_d_pres",

        "dm1_disruptions_rec_num",
        "dm2_disruptions_rec_num",
        "dm3_disruptions_rec_num",
        "prev_7_d_disruptions_rec_num",
        "prev_30_d_disruptions_rec_num",
        "career_disruptions_rec_per_cal_d",
        "career_disruptions_rec_per_d_pres",

        "dm1_sacrifices_rec_num",
        "dm2_sacrifices_rec_num",
        "dm3_sacrifices_rec_num",
        "prev_7_d_sacrifices_rec_num",
        "prev_30_d_sacrifices_rec_num",
        "career_sacrifices_rec_per_cal_d",
        "career_sacrifices_rec_per_d_pres",

        "dm1_sabotages_rec_num",
        "dm2_sabotages_rec_num",
        "dm3_sabotages_rec_num",
        "prev_7_d_sabotages_rec_num",
        "prev_30_d_sabotages_rec_num",
        "career_sabotages_rec_per_cal_d",
        "career_sabotages_rec_per_d_pres",
        ]] = None

    # This defragments the DF and eliminates fragmentation warnings.
    cfg.pers_day_df = cfg.pers_day_df.copy()

    # ------------------------------------------------------------------
    # (Potential) target columns.
    # ------------------------------------------------------------------

    cfg.potential_target_columns = [
        "dp1_resignation_yn",
        "dp1_termination_yn",
        "dp1_presence_yn",
        "dp1_absence_yn",
        "dp1_eff_rec_val",
        "dp1_ideas_rec_num",
        "dp1_lapses_rec_num",
        "dp1_feats_rec_num",
        "dp1_slips_rec_num",
        "dp1_teamworks_rec_num",
        "dp1_disruptions_rec_num",
        "dp1_sacrifices_rec_num",
        "dp1_sabotages_rec_num",

        "nxt_7_d_resignation_yn",
        "nxt_7_d_termination_yn",
        "nxt_7_d_presences_num",
        "nxt_7_d_absences_num",
        "nxt_7_d_ideas_rec_num",
        "nxt_7_d_lapses_rec_num",
        "nxt_7_d_feats_rec_num",
        "nxt_7_d_slips_rec_num",
        "nxt_7_d_teamworks_rec_num",
        "nxt_7_d_disruptions_rec_num",
        "nxt_7_d_sacrifices_rec_num",
        "nxt_7_d_sabotages_rec_num",
        "nxt_7_d_eff_rec_mean",
        "nxt_7_d_eff_rec_sd",

        "nxt_30_d_resignation_yn",
        "nxt_30_d_termination_yn",
        "nxt_30_d_presences_num",
        "nxt_30_d_absences_num",
        "nxt_30_d_ideas_rec_num",
        "nxt_30_d_lapses_rec_num",
        "nxt_30_d_feats_rec_num",
        "nxt_30_d_slips_rec_num",
        "nxt_30_d_teamworks_rec_num",
        "nxt_30_d_disruptions_rec_num",
        "nxt_30_d_sacrifices_rec_num",
        "nxt_30_d_sabotages_rec_num",
        "nxt_30_d_eff_rec_mean",
        "nxt_30_d_eff_rec_sd"
        ]

    cfg.pers_day_df[cfg.potential_target_columns] = None

    # This defragments the DF and eliminates fragmentation warnings.
    cfg.pers_day_df = cfg.pers_day_df.copy()


def engineer_pers_day_df_features_and_targets():
    """
    Updates cfg.pers_day_df to include a diverse range of features and
    targets for possible use by machine-learning algorithms. Each row
    in cfg.pers_day_df will represent a particular day's available
    feature input (and targets to be predicted and assessed against
    the actual values) for a particular subject.
    """

    print("Beginning engineer_pers_day_df_features_and_targets_from_behavs_act_df().")

    # It's not clear why this needs to be converted to a string,
    # and then reconvert back into PD datetime format. It seems to have
    # become a numpy.datetime64 somewhere along the way.
    cfg.pers_day_df["d0_date"] = cfg.pers_day_df["d0_date"].astype(str)
    for i in range(len(cfg.pers_day_df)):
        cfg.pers_day_df["d0_date"].values[i] = \
            (pd.Timestamp(cfg.pers_day_df["d0_date"].values[i])).to_pydatetime()
        cfg.pers_day_df.loc[i, "d0_day_of_month_num"] = \
            cfg.behavs_act_df["d0_date"].values[1].day
        cfg.pers_day_df.loc[i, "d0_month_of_year_num"] = \
            cfg.behavs_act_df["d0_date"].values[1].month

    # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
    # ● Populate the columns already created in cfg.pers_day_df.
    # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

    # For each unique person who appears as a subject in the dataset...
    for pers in cfg.pers_day_df["sub_ID"].unique().tolist():

        # ... create a temporary trimmed copy of cfg.pers_day_df that only
        # has entries for that subject. This temp df will be used to perform
        # calculations that will populate columns in cfg.pers_day_df.
        sub_ID_temp_df = cfg.pers_day_df.copy()
        sub_ID_temp_df = sub_ID_temp_df[ sub_ID_temp_df["sub_ID"] == pers ]

        # Now step through each of the rows in cfg.pers_day_df that has this
        # person as its subject. For each of those rows, use sub_ID_temp_df to
        # perform calculations that will populate fields in that row
        # of cfg.pers_day_df.

        for i in range(len(sub_ID_temp_df)):

            # Find the corresponding row index in cfg.pers_day_df
            idx_pers_day_df = (sub_ID_temp_df.iloc[[i]]).index[0]

            # Calculate the date of D0.
            d0_date = sub_ID_temp_df["d0_date"].values[i]

            # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
            # █ Calculate features.
            # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
            # ● Calculate features relating to day D-1.
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

            # Calculate the date of D-1.
            dm1_date = d0_date - timedelta(days=1)

            # ----------------------------------------------------------
            # Search for a row in sub_ID_temp_df where "d0_date" == 
            # dm1_date and copy its values into cfg.pers_day_df.
            # ----------------------------------------------------------
            sub_ID_temp_df_desired_day \
                = sub_ID_temp_df[ sub_ID_temp_df["d0_date"] == dm1_date]
            if len(sub_ID_temp_df_desired_day) > 0:
                idx = ( sub_ID_temp_df_desired_day.iloc[[0]] ).index[0]

                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_eff_rec_val"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "d0_eff_rec_val"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_presence_yn"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Presence"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_ideas_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Idea"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_lapses_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Lapse"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_feats_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Feat"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_slips_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Slip"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_disruptions_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Disruption"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_sacrifices_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sacrifice"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm1_sabotages_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sabotage"]

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
            # ● Calculate features relating to day D-2.
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

            # Calculate the date of D-1.
            dm2_date = d0_date - timedelta(days=2)

            # ----------------------------------------------------------
            # Search for a row in sub_ID_temp_df where "d0_date" == 
            # dm2_date and copy its values into cfg.pers_day_df.
            # ----------------------------------------------------------
            sub_ID_temp_df_desired_day = \
                sub_ID_temp_df[ sub_ID_temp_df["d0_date"] == dm2_date]
            if len(sub_ID_temp_df_desired_day) > 0:
                idx = ( sub_ID_temp_df_desired_day.iloc[[0]] ).index[0]

                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_eff_rec_val"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "d0_eff_rec_val"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_ideas_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Idea"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_lapses_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Lapse"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_feats_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Feat"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_slips_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Slip"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_disruptions_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Disruption"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_sacrifices_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sacrifice"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm2_sabotages_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sabotage"]

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 
            # ● Calculate features relating to day D-3.
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 

            # Calculate the date of D-3.
            dm3_date = d0_date - timedelta(days=3)

            # ----------------------------------------------------------
            # Search for a row in sub_ID_temp_df where "d0_date" == 
            # dm3_date and copy its values into cfg.pers_day_df.
            # ----------------------------------------------------------
            sub_ID_temp_df_desired_day \
                = sub_ID_temp_df[ sub_ID_temp_df["d0_date"] == dm3_date]
            if len(sub_ID_temp_df_desired_day) > 0:
                idx = ( sub_ID_temp_df_desired_day.iloc[[0]] ).index[0]

                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_eff_rec_val"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "d0_eff_rec_val"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_ideas_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Idea"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_lapses_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Lapse"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_feats_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Feat"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_slips_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Slip"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_disruptions_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Disruption"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_sacrifices_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sacrifice"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dm3_sabotages_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sabotage"]

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # ● Calculate features relating to day D-4.
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D-4.
            dm4_date = d0_date - timedelta(days=4)

            # ----------------------------------------------------------
            # Search for a row in sub_ID_temp_df where "d0_date" == 
            # dm4_date and copy its values into cfg.pers_day_df.
            # ----------------------------------------------------------
            sub_ID_temp_df_desired_day \
                = sub_ID_temp_df[ sub_ID_temp_df["d0_date"] == dm4_date]
            if len(sub_ID_temp_df_desired_day) > 0:
                idx = ( sub_ID_temp_df_desired_day.iloc[[0]] ).index[0]

                cfg.pers_day_df.loc[idx_pers_day_df, "dm4_eff_rec_val"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "d0_eff_rec_val"]

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # ● Calculate features relating to the 7 previous days, including 
            # the current day (i.e., D-6 through D0).
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D-6.
            dm6_date = d0_date - timedelta(days=6)

            # If the earliest date in sub_ID_temp_df for this person falls
            # after the D-6 date, then there's not a long enough span of data 
            # to create a timed DF that ends on the D0 date. If there *is* a 
            # long enough data range, create a DF of entries running from D-6 
            # through D0.
            if sub_ID_temp_df["d0_date"].min() <= dm6_date:
                sub_ID_temp_df_trmmd = sub_ID_temp_df.copy()
                sub_ID_temp_df_trmmd = \
                    sub_ID_temp_df_trmmd[ \
                        (sub_ID_temp_df_trmmd["d0_date"] >= dm6_date) \
                        & (sub_ID_temp_df_trmmd["d0_date"] <= d0_date) 
                        ]

                # Attendance.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_presences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Presence"].sum()
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_absences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Absence"].sum()

                # Efficacy.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_eff_rec_mean"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].mean()
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_eff_rec_sd"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].std()

                # Ideas.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_ideas_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Idea"].sum()

                # Lapses.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_lapses_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum()

                # Feats.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_feats_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Feat"].sum()

                # Slips.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_slips_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Slip"].sum()

                # Teamworks.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_teamworks_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum()

                # Disruptions.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_disruptions_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum()

                # Sacrifices.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_sacrifices_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum()

                # Sabotages.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_7_d_sabotages_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum()

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # ● Calculate features relating to the 30 previous days, 
            # including the current day (i.e., D-29 through D0).
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D-29.
            dm29_date = d0_date - timedelta(days=29)

            # If the earliest date in behavs_act_temp_df trimmed for this 
            # person falls after the D-29 date, then there's not a long enough 
            # span of data to create a timed DF that ends on the D0 date. If 
            # there *is* a long enough data range, create a DF of entries 
            # running from D-29 through D0.
            if sub_ID_temp_df["d0_date"].min() <= dm29_date:
                sub_ID_temp_df_trmmd = sub_ID_temp_df.copy()
                sub_ID_temp_df_trmmd = \
                    sub_ID_temp_df_trmmd[ \
                        (sub_ID_temp_df_trmmd["d0_date"] >= dm29_date) \
                        & (sub_ID_temp_df_trmmd["d0_date"] <= d0_date) 
                        ]

                # Attendance.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_presences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Presence"].sum()
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_absences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Absence"].sum()

                # Efficacy.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_eff_rec_mean"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].mean()
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_eff_rec_sd"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].std()

                # Ideas.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_ideas_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Idea"].sum()

                # Lapses.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_lapses_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum()

                # Feats.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_feats_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Feat"].sum()

                # Slips.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_slips_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Slip"].sum()

                # Teamworks.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_teamworks_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum()

                # Disruptions.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_disruptions_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum()

                # Sacrifices.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_sacrifices_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum()

                # Sabotages.
                cfg.pers_day_df.loc[idx_pers_day_df, "prev_30_d_sabotages_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum()


            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # Calculate mean "career" features that are based on a 
            # person's entire career, regardless of how short or long it 
            # might be.
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            sub_ID_temp_df_trmmd = sub_ID_temp_df.copy()
            sub_ID_temp_df_trmmd = \
                sub_ID_temp_df_trmmd[ sub_ID_temp_df_trmmd["d0_date"] <= d0_date ]

            # Calculate the total number of calendar days covered by this 
            # career span.
            days_cal = (sub_ID_temp_df_trmmd["d0_date"].max() \
                - sub_ID_temp_df_trmmd["d0_date"].min() ).days + 1 

            # Calculate the total number of days on which the person was 
            # present during this career span.
            days_prsnt = sub_ID_temp_df_trmmd["rcomp_Presence"].sum()

            # Attendance.
            cfg.pers_day_df.loc[idx_pers_day_df, "career_presences_per_cal_d"] \
                = days_prsnt / days_cal
            cfg.pers_day_df.loc[idx_pers_day_df, "career_absences_per_cal_d"] \
                = sub_ID_temp_df_trmmd["rcomp_Absence"].sum() / days_cal

            # Efficacy.
            cfg.pers_day_df.loc[idx_pers_day_df, "career_eff_rec_mean"] \
                = sub_ID_temp_df_trmmd["d0_eff_rec_val"].mean()
            cfg.pers_day_df.loc[idx_pers_day_df, "career_eff_rec_sd"] \
                = sub_ID_temp_df_trmmd["d0_eff_rec_val"].std()

            # Ideas.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_ideas_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_ideas_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Idea"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_ideas_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Idea"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_ideas_rec_per_d_pres"] \
                        = None

            # Lapses.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_lapses_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_lapses_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_lapses_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_lapses_rec_per_d_pres"] \
                        = None

            # Feats.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_feats_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_feats_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Feat"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_feats_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Feat"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_feats_rec_per_d_pres"] \
                        = None

            # Slips.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_slips_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_slips_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Slip"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_slips_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Slip"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_slips_rec_per_d_pres"] \
                        = None

            # Teamworks.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_teamworks_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_teamworks_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_teamworks_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_teamworks_rec_per_d_pres"] \
                        = None

            # Disruptions.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_disruptions_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_disruptions_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_disruptions_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_disruptions_rec_per_d_pres"] \
                        = None

            # Sacrifices.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sacrifices_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_sacrifices_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sacrifices_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sacrifices_rec_per_d_pres"] \
                        = None

            # Sabotages.
            if isinstance(
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sabotages_rec_per_d_pres"],
                    float
                    ):
                cfg.pers_day_df.loc[idx_pers_day_df, "career_sabotages_rec_per_cal_d"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum() / days_cal
                if days_prsnt > 0:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sabotages_rec_per_d_pres"] \
                        = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum() / days_prsnt
                else:
                    cfg.pers_day_df.loc[idx_pers_day_df, "career_sabotages_rec_per_d_pres"] \
                        = None

            # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
            # █ Calculate targets.
            # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # Calculate targets relating to the immediately following 
            # day (D+1).
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D+1.
            dp1_date = d0_date + timedelta(days=1)

            sub_ID_temp_df_desired_day \
                = sub_ID_temp_df[ sub_ID_temp_df["d0_date"] == dp1_date]
            if len(sub_ID_temp_df_desired_day) > 0:
                idx = ( sub_ID_temp_df_desired_day.iloc[[0]] ).index[0]

                # Resignation.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_resignation_yn"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Resignation"]

                # Termination.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_termination_yn"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Termination"]

                # Attendance.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_presence_yn"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Presence"]
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_absence_yn"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Absence"]

                # Efficacy.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_eff_rec_val"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "d0_eff_rec_val"]

                # Ideas.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_ideas_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Idea"]

                # Lapses.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_lapses_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Lapse"]

                # Feats.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_feats_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Feat"]

                # Slips.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_slips_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Slip"]

                # Teamworks.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_teamworks_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Teamwork"]

                # Disruptions.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_disruptions_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Disruption"]

                # Sacrifices.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_sacrifices_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sacrifice"]

                # Sabotages.
                cfg.pers_day_df.loc[idx_pers_day_df, "dp1_sabotages_rec_num"] \
                    = sub_ID_temp_df_desired_day.loc[idx, "rcomp_Sabotage"]

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # ● Calculate targets relating to the following 7 days 
            # ● (i.e., D+1 through D+7).
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D+7.
            dp7_date = d0_date + timedelta(days=7)

            # If the latest date in sub_ID_temp_df for this person falls
            # before the D+7 date, then there's not a long enough span of data 
            # to create a timed DF that ends on the D+7 date. If there *is* a 
            # long enough data range, create a DF of entries running from D+1 
            # through D+7.
            if sub_ID_temp_df["d0_date"].max() >= dp7_date:
                sub_ID_temp_df_trmmd = sub_ID_temp_df.copy()
                sub_ID_temp_df_trmmd = \
                    sub_ID_temp_df_trmmd[ \
                        (sub_ID_temp_df_trmmd["d0_date"] >= dp1_date) \
                        & (sub_ID_temp_df_trmmd["d0_date"] <= dp7_date)
                        ]

                # Resignation.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_resignation_yn"] \
                    = sub_ID_temp_df_trmmd["rcomp_Resignation"].sum()

                # Termination.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_termination_yn"] \
                    = sub_ID_temp_df_trmmd["rcomp_Termination"].sum()

                # Attendance.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_presences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Presence"].sum()
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_absences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Absence"].sum()

                # Efficacy.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_eff_rec_mean"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].mean()
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_eff_rec_sd"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].std()

                # Ideas.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_ideas_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Idea"].sum()

                # Lapses.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_lapses_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum()

                # Feats.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_feats_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Feat"].sum()

                # Slips.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_slips_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Slip"].sum()

                # Teamworks.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_teamworks_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum()

                # Disruptions.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_disruptions_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum()

                # Sacrifices.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_sacrifices_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum()

                # Sabotages.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_7_d_sabotages_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum()

            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
            # ● Calculate targets relating to the following 30 days 
            # ● (i.e., D+1 through D+30).
            # ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●

            # Calculate the date of D+30.
            dp30_date = d0_date + timedelta(days=30)

            # If the latest date in sub_ID_temp_df for this person falls
            # before the D+30 date, then there's not a long enough span of 
            # data to create a timed DF that ends on the D+30 date. If there 
            # *is* a long enough data range, create a DF of entries running 
            # from D+1 through D+30.
            if sub_ID_temp_df["d0_date"].max() >= dp30_date:
                sub_ID_temp_df_trmmd = sub_ID_temp_df.copy()
                sub_ID_temp_df_trmmd = \
                    sub_ID_temp_df_trmmd[ \
                        (sub_ID_temp_df_trmmd["d0_date"] >= dp1_date) \
                        & (sub_ID_temp_df_trmmd["d0_date"] <= dp30_date)
                        ]

                # Resignation.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_resignation_yn"] \
                    = sub_ID_temp_df_trmmd["rcomp_Resignation"].sum()

                # Termination.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_termination_yn"] \
                    = sub_ID_temp_df_trmmd["rcomp_Termination"].sum()

                # Attendance.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_presences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Presence"].sum()
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_absences_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Absence"].sum()

                # Efficacy.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_eff_rec_mean"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].mean()
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_eff_rec_sd"] \
                    = sub_ID_temp_df_trmmd["d0_eff_rec_val"].std()

                # Ideas.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_ideas_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Idea"].sum()

                # Lapses.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_lapses_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Lapse"].sum()

                # Feats.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_feats_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Feat"].sum()

                # Slips.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_slips_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Slip"].sum()

                # Teamworks.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_teamworks_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Teamwork"].sum()

                # Disruptions.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_disruptions_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Disruption"].sum()

                # Sacrifices.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_sacrifices_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sacrifice"].sum()

                # Sabotages.
                cfg.pers_day_df.loc[idx_pers_day_df, "nxt_30_d_sabotages_rec_num"] \
                    = sub_ID_temp_df_trmmd["rcomp_Sabotage"].sum()


def handle_dataset_rows_with_null_values():
    """
    Handles dataset rows that contain Null values, to prepare the data
    for use with models.
    """

    # For some features/targets, Null values can be replaced with, e.g.,
    # 0 or -1.
    cols_to_check = [
        # Note! The two values below should eventually be calculated, 
        # not ignored.
        #"d0_day_of_month_num",
        #"d0_month_of_year_num",
        "d0_eff_rec_val",
        "dm1_eff_rec_val",
        "dm2_eff_rec_val",
        "dm3_eff_rec_val",
        "dm4_eff_rec_val",
        "dm1_presence_yn",
        "dm1_absence_yn",
        "dm1_ideas_rec_num",
        "dm2_ideas_rec_num",
        "dm3_ideas_rec_num",
        "dm1_lapses_rec_num",
        "dm2_lapses_rec_num",
        "dm3_lapses_rec_num",
        "dm1_feats_rec_num",
        "dm2_feats_rec_num",
        "dm3_feats_rec_num",
        "dm1_slips_rec_num",
        "dm2_slips_rec_num",
        "dm3_slips_rec_num",
        "dm1_teamworks_rec_num",
        "dm2_teamworks_rec_num",
        "dm3_teamworks_rec_num",
        "dm1_disruptions_rec_num",
        "dm2_disruptions_rec_num",
        "dm3_disruptions_rec_num",
        "dm1_sacrifices_rec_num",
        "dm2_sacrifices_rec_num",
        "dm3_sacrifices_rec_num",
        "dm1_sabotages_rec_num",
        "dm2_sabotages_rec_num",
        "dm3_sabotages_rec_num",
        ]

    cfg.pers_day_df.update(cfg.pers_day_df[cols_to_check].fillna(-1))

    # For other features/targets, a Null value should cause the affected
    # row to be deleted.
    cols_to_check = [
        "prev_7_d_eff_rec_mean",
        "prev_7_d_eff_rec_sd",
        "prev_30_d_eff_rec_mean",
        "prev_30_d_eff_rec_sd",
        "prev_7_d_presences_num",
        "prev_30_d_presences_num",
        "prev_7_d_absences_num",
        "prev_30_d_absences_num",
        "prev_7_d_ideas_rec_num",
        "prev_30_d_ideas_rec_num",
        "prev_7_d_lapses_rec_num",
        "prev_30_d_lapses_rec_num",
        "prev_7_d_feats_rec_num",
        "prev_30_d_feats_rec_num",
        "prev_7_d_slips_rec_num",
        "prev_30_d_slips_rec_num",
        "prev_7_d_teamworks_rec_num",
        "prev_30_d_teamworks_rec_num",
        "prev_7_d_disruptions_rec_num",
        "prev_30_d_disruptions_rec_num",
        "prev_7_d_sacrifices_rec_num",
        "prev_30_d_sacrifices_rec_num",
        "prev_7_d_sabotages_rec_num",
        "prev_30_d_sabotages_rec_num",
        ]
    cols_to_check.extend(cfg.potential_target_columns)
    # Note that excluding rows with a Null value in *any* potential
    # target column may exclude more rows than is necessary, given the
    # target columns that will *actually* be used for a given modelling
    # sequence.

    for col in cols_to_check:
        cfg.pers_day_df = cfg.pers_day_df[ cfg.pers_day_df[col].notna() ]


def create_train_valid_test_split_from_pers_day_df(
    valid_prtn_of_sub_IDs_u,
    test_prtn_of_sub_IDs_u,
    ):
    """
    Creates DataFrames with training, validation, and test data.

    PARAMETERS
    ----------
    valid_prtn_of_sub_IDs_u
        The portion of the total number of subjects to use for validation
    test_prtn_of_sub_IDs_u
        The portion of the total number of subjects to use for testing
    """

    # Create a list of all the persons who appear as subjects in the 
    # observations.
    sub_ID_values_list = cfg.pers_day_df["sub_ID"].unique().tolist()

    # Randomly shuffle that list.
    random.Random(cfg.USER_CONFIGURABLE_MODEL_SETTING_C).shuffle(sub_ID_values_list)

    # The now-shuffled list of subject IDs will be split into training, 
    # validation, and test groups, with each person (and all of his 
    # entries) falling into exactly one group.
    sub_ID_unique_values_num = len(sub_ID_values_list)
    train_prtn_of_sub_IDs \
        = 1.0 - valid_prtn_of_sub_IDs_u - test_prtn_of_sub_IDs_u
    train_group_final_member_index \
        = int(train_prtn_of_sub_IDs * sub_ID_unique_values_num)
    valid_set_final_row_index = train_group_final_member_index + \
        int(valid_prtn_of_sub_IDs_u * sub_ID_unique_values_num)

    sub_IDs_train_group_list \
        = sub_ID_values_list[0:train_group_final_member_index]
    sub_IDs_valid_group_list \
        = sub_ID_values_list[train_group_final_member_index:valid_set_final_row_index]
    sub_IDs_test_group_list \
        = sub_ID_values_list[valid_set_final_row_index:]

    cfg.Xy_train_df = cfg.pers_day_df.copy()
    cfg.Xy_train_df \
        = cfg.Xy_train_df[ cfg.Xy_train_df["sub_ID"].isin(sub_IDs_train_group_list)  ]

    cfg.Xy_valid_df = cfg.pers_day_df.copy()
    cfg.Xy_valid_df \
        = cfg.Xy_valid_df[ cfg.Xy_valid_df["sub_ID"].isin(sub_IDs_valid_group_list)  ]

    cfg.Xy_test_df = cfg.pers_day_df.copy()
    cfg.Xy_test_df \
        = cfg.Xy_test_df[ cfg.Xy_test_df["sub_ID"].isin(sub_IDs_test_group_list)  ]

    print(
        "Number of subject IDs in training group: ",
        len(sub_IDs_train_group_list)
        )
    print(
        "Number of subject IDs in validation group: ",
        len(sub_IDs_valid_group_list)
        )
    print(
        "Number of subject IDs in test group: ",
        len(sub_IDs_test_group_list)
        )
    print("sub_ID_unique_values_num: ", sub_ID_unique_values_num)


def select_features_and_targets_in_pers_day_df(
    features_list_non_OHE_u,
    features_list_OHE_u,
    targets_list_u,
    feature_to_use_for_naive_persistence_model_u,
    feature_to_use_for_naive_mean_model_u,
    feature_for_calculating_target_range_u,
    ):
    """
    Extracts columns with the selected features and target(s) from the
    training, validation, and test datasets.

    PARAMETERS
    ----------
    features_list_non_OHE_u
        A list of any non-OHE features to use with models
    features_list_OHE_u
        A list of any OHE features to use with models (referenced by 
        OHE prefix)
    targets_list_u
        A list of any targets to use with models
    feature_to_use_for_naive_persistence_model_u
        The feature for the Naive Persistence model to return
    feature_to_use_for_naive_mean_model_u
        The feature for the Naive Mean model to return
    feature_for_calculating_target_range_u
        The feature to guide calculation of the target's likely range
    """
    # ------------------------------------------------------------------
    # Extract feature columns.
    # ------------------------------------------------------------------

    list_of_OHE_feature_columns = []
    for prefix in features_list_OHE_u:
        list_of_OHE_feature_columns_this_prefix \
            = [col for col in cfg.Xy_test_df if col.startswith(prefix)]

        if len(list_of_OHE_feature_columns) == 0:
            list_of_OHE_feature_columns \
                = list_of_OHE_feature_columns_this_prefix
        else:
            # "Extend" modifies the original list in place.
            list_of_OHE_feature_columns.extend(list_of_OHE_feature_columns_this_prefix)

    # "Extend" modifies the original list in place.
    features_list_non_OHE_u.extend(list_of_OHE_feature_columns)
    list_of_feature_columns = features_list_non_OHE_u
    print("list_of_feature_columns: ", list_of_feature_columns)

    cfg.X_train_slctd_df = cfg.Xy_train_df[list_of_feature_columns]
    cfg.X_train_slctd_df = cfg.X_train_slctd_df.reset_index(drop=True)

    cfg.X_valid_slctd_df = cfg.Xy_valid_df[list_of_feature_columns]
    cfg.X_valid_slctd_df = cfg.X_valid_slctd_df.reset_index(drop=True)

    # These are currently omitted, to reduce memory usage.
    #cfg.X_test_slctd_df = cfg.Xy_test_df[list_of_feature_columns]
    #cfg.X_test_slctd_df = cfg.X_test_slctd_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Extract target columns.
    # ------------------------------------------------------------------

    print("list_of_target_columns: ", targets_list_u)

    cfg.y_train_slctd_df = cfg.Xy_train_df[targets_list_u]
    cfg.y_train_slctd_df = cfg.y_train_slctd_df.reset_index(drop=True)

    cfg.y_valid_slctd_df = cfg.Xy_valid_df[targets_list_u]
    cfg.y_valid_slctd_df = cfg.y_valid_slctd_df.reset_index(drop=True)

    # These are currently omitted, to reduce memory usage.
    #cfg.y_test_slctd_df = cfg.Xy_test_df[targets_list_u]
    #cfg.y_test_slctd_df = cfg.y_test_slctd_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Set Naive Persistence and Naive Mean model features to return.
    # ------------------------------------------------------------------

    cfg.feature_to_return_naive_persistence \
        = feature_to_use_for_naive_persistence_model_u
    cfg.feature_to_return_naive_mean = feature_to_use_for_naive_mean_model_u

    # ------------------------------------------------------------------
    # Set feature(s) for use in calculating target values' likely range.
    # ------------------------------------------------------------------

    cfg.feature_for_calculating_target_range \
        = feature_for_calculating_target_range_u


def preprocess_data_for_use_with_all_models():
    """
    Preprocesses the training, validation, and test data for use with
    all simple (sklearn-type) and deep-learning (keras-type) models.
    """

    # Without any preprocessing.
    cfg.X_train_slctd_df_preprocessed = cfg.X_train_slctd_df
    cfg.X_valid_slctd_df_preprocessed = cfg.X_valid_slctd_df

    # These options are not currently used.
    #cfg.X_train_slctd_df_preprocessed = preprocessing.StandardScaler(cfg.X_train_slctd_df)
    #cfg.X_valid_slctd_df_preprocessed = preprocessing.StandardScaler(cfg.X_valid_slctd_df)

    # These options are not currently used.
    #scaler_curr = preprocessing.StandardScaler()
    #scaler_curr.fit(cfg.X_train_slctd_df)
    #cfg.X_train_slctd_df_preprocessed = scaler_curr.transform(cfg.X_train_slctd_df)
    #cfg.X_valid_slctd_df_preprocessed = scaler_curr.transform(cfg.X_valid_slctd_df)


def create_df_for_tracking_model_metrics():
    """
    Creates a DataFrame for tracking the metrics of all models.
    """

    cfg.models_metrics_df = pd.DataFrame(columns=[
        "model_id",
        "model_type", # "Base Target", "Ceiling", "Floor", or "Joint Range"
        "model_parameters_description",

        # Relevant for the Base Target Model.
        "MAE",
        "MSE",

        # Relevant for a Ceiling Model.
        "PATGTC",
        "AMORPDAC",
        "AMIRPDBC",
        "OCE",

        # Relevant for a Floor Model.
        "PATLTF",
        "AMORPDBF",
        "AMIRPDAF",
        "OFE",

        # Relevant for a Joint Range Model.
        "MPRS",
        "IRP",
        "PATOR",
        "MSADRE",
        "ORP",
        ])


def train_and_validate_base_target_model(
    ):
    """
    This model attempts to predict the actual target value itself (not 
    the ceiling or floor of the range).
    """

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = "BT"

    base_target_model = RandomForestRegressor(
        n_estimators=300,
        random_state=cfg.USER_CONFIGURABLE_MODEL_SETTING_C)

    # This puts the y-value dat in the necessary format for sklearn's RF
    # model.
    y_train_slctd_df_this = np.ravel(cfg.y_train_slctd_df.astype("float"))

    base_target_model.fit(
        cfg.X_train_slctd_df_preprocessed,
        y_train_slctd_df_this
        )

    cfg.y_preds_base_target_curr \
        = base_target_model.predict(cfg.X_valid_slctd_df_preprocessed)

    # Calculate the largest calculated ±y-value associated with this 
    # model, for use in setting the y-max in Joint Range plots.
    y_max_candidates_for_plotting_list = list()
    for target_pred, target_actual in zip(
            cfg.y_preds_base_target_curr,
            cfg.y_valid_slctd_df.iloc[:,0].tolist()
            ):
        y_max_candidates_for_plotting_list.append(abs(target_pred - target_actual))
        y_max_candidates_max = max(y_max_candidates_for_plotting_list)
    cfg.models_info_dict[cfg.model_id_curr]["y_max_candidate_for_joint_range_plots"] \
        = y_max_candidates_max

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        cfg.y_preds_base_target_curr,
        )
    vis.plot_model_results_scatter(
        cfg.y_preds_base_target_curr,
        cfg.y_valid_slctd_df,
        )

    # Create a DF of the X_valid features with the base target y_preds
    # concatenated to it.
    y_preds_base_target_curr_df \
        = pd.DataFrame(cfg.y_preds_base_target_curr, columns=["y_preds_base_target"])

    cfg.X_valid_slctd_df_preprocessed_with_y_preds_base = pd.merge(
        left = cfg.X_valid_slctd_df_preprocessed,
        right = pd.DataFrame(y_preds_base_target_curr_df),
        left_index=True,
        right_index=True,
        )

    # This is for exporting to a pickle file.
    y_preds_base_target_curr_this = cfg.y_preds_base_target_curr.copy()
    if isinstance(y_preds_base_target_curr_this, pd.DataFrame):
        y_preds_base_target_curr_this \
            = y_preds_base_target_curr_this.iloc[:,0].tolist()
    if isinstance(y_preds_base_target_curr_this, np.ndarray):
        y_preds_base_target_curr_this = y_preds_base_target_curr_this.tolist()
        y_preds_base_target_curr_this = np.ravel(y_preds_base_target_curr_this)
    cfg.models_info_dict["BT"]["y_preds_list"] = y_preds_base_target_curr_this


def train_and_validate_nn_ceiling_model(
    lowest_safe_ceiling_loss_func_id_u
    ):
    """
    This model attempts to predict the ceiling for the prediction
    interval that's as low as possible while still being high enough
    to be greater than the actual target value.

    PARAMETERS
    ----------
    lowest_safe_ceiling_loss_func_id_u
        ID of the lowest safe ceiling loss function that should be used
        in training the model
    """

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = lowest_safe_ceiling_loss_func_id_u

    cfg.lowest_safe_ceiling_loss_func_id \
        = lowest_safe_ceiling_loss_func_id_u

    keras.backend.clear_session()

    base_target_model = Sequential()
    base_target_model.add(Dense(
        7, activation="relu",
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        ))
    base_target_model.add(Dense(
        4, activation="relu",
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        ))
    base_target_model.add(Dense(
        1, activation="relu",
        ))
    base_target_model.compile(
        loss=lowest_safe_ceiling_loss_func,
        optimizer="adam",
        metrics=["accuracy"]
        )

    y_train_slctd_df_this = cfg.y_train_slctd_df

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=35)

    base_target_model.fit(
        np.asarray(cfg.X_train_slctd_df_preprocessed).astype('float32'),
        np.asarray(y_train_slctd_df_this).astype('float32'),
        verbose=0,
        epochs=800,
        callbacks=[early_stopping]
        )

    y_preds_ceiling_curr \
        = base_target_model.predict(
            np.asarray(cfg.X_valid_slctd_df_preprocessed).astype('float32'),
            )
    y_preds_ceiling_list = y_preds_ceiling_curr.tolist()
    y_preds_ceiling_list \
        = [ceiling[0] if ceiling[0] > 0 else 0 for ceiling in y_preds_ceiling_list]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_ceiling_list

    # Calculate the largest calculated ±y-value associated with this 
    # model, for use in setting the y-max in Joint Range plots.
    y_max_candidates_for_plotting_list = list()
    for predicted_ceiling, predicted_target in zip(
            y_preds_ceiling_list,
            cfg.y_preds_base_target_curr,
            ):
        y_max_candidates_for_plotting_list.append(abs(predicted_ceiling - predicted_target))
        y_max_candidates_max = max(y_max_candidates_for_plotting_list)
    cfg.models_info_dict[cfg.model_id_curr]["y_max_candidate_for_joint_range_plots"] \
        = y_max_candidates_max

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_ceiling_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_ceiling_list,
        cfg.y_valid_slctd_df,
        "ceiling"
        )


def train_and_validate_nn_floor_model(
    highest_safe_floor_loss_func_id_u
    ):
    """
    This model attempts to predict the floor for the prediction
    interval that's as high as possible while still being low enough
    to be greater than the actual target value.

    PARAMETERS
    ----------
    highest_safe_floor_loss_func_id_u
        ID of the highest safe floor loss function that should be used
        in training the model
    """

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = highest_safe_floor_loss_func_id_u

    cfg.highest_safe_floor_loss_func_id = highest_safe_floor_loss_func_id_u

    keras.backend.clear_session()

    base_target_model = Sequential()
    base_target_model.add(Dense(
        7, activation="relu",
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        ))
    base_target_model.add(Dense(
        4, activation="relu",
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        ))
    base_target_model.add(Dense(
        1, activation="relu",
        ))
    base_target_model.compile(
        loss=highest_safe_floor_loss_func,
        optimizer="adam",
        metrics=["accuracy"]
        )

    y_train_slctd_df_this = cfg.y_train_slctd_df

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=35)

    base_target_model.fit(
        np.asarray(cfg.X_train_slctd_df_preprocessed).astype('float32'),
        np.asarray(y_train_slctd_df_this).astype('float32'),
        verbose=0,
        epochs=800,
        callbacks=[early_stopping]
        )

    y_preds_floor_curr \
        = base_target_model.predict(
            np.asarray(cfg.X_valid_slctd_df_preprocessed).astype('float32'),
            )
    y_preds_floor_list = y_preds_floor_curr.tolist()
    y_preds_floor_list \
        = [floor[0] if floor[0] > 0 else 0 for floor in y_preds_floor_curr]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_floor_list

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_floor_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_floor_list,
        cfg.y_valid_slctd_df,
        "floor"
        )


def generate_metrics_for_model(
    y_actual_for_metrics_plots_u,
    y_preds_for_metrics_plots_u,
    ):
    """
    Generates relevant metrics for the given model.

    PARAMETERS
    ----------
    y_actual_for_metrics_plots_u
        The actual (typically target) values to be used in calculating
        metrics
    y_preds_for_metrics_plots_u
        The predicted (e.g., target, ceiling, or floor) values to be
        used in calculating metrics
    """

    model_type = cfg.models_info_dict[cfg.model_id_curr]["model_type"]

    # Deal with the fact that different models produce their output
    # in different formats (pdDataFrame, np.ndarray, list, etc.).
    if isinstance(y_actual_for_metrics_plots_u, pd.DataFrame):
        y_actual_for_metrics_plots_u \
            = y_actual_for_metrics_plots_u.iloc[:,0].tolist()
    if isinstance(y_preds_for_metrics_plots_u, np.ndarray):
        y_preds_for_metrics_plots_u = y_preds_for_metrics_plots_u.tolist()
        y_preds_for_metrics_plots_u = np.ravel(y_preds_for_metrics_plots_u)

    y_preds_base_target_curr_this = cfg.y_preds_base_target_curr.copy()
    if isinstance(y_preds_base_target_curr_this, pd.DataFrame):
        y_preds_base_target_curr_this \
            = y_preds_base_target_curr_this.iloc[:,0].tolist()
    if isinstance(y_preds_base_target_curr_this, np.ndarray):
        y_preds_base_target_curr_this = y_preds_base_target_curr_this.tolist()
        y_preds_base_target_curr_this = np.ravel(y_preds_base_target_curr_this)

    # ==================================================================
    # Reset metrics.
    # ==================================================================

    # By default, all values will be None, unless they're overwritten
    # by a particular case below (e.g., for Base Target, Ceiling,
    # or Floor Models).

    cfg.mae = None
    cfg.mse = None

    cfg.patgtc = None
    cfg.amorpdac = None
    cfg.amirpdbc = None
    cfg.oce = None

    cfg.patltf = None
    cfg.amorpdbf = None
    cfg.amirpdaf = None
    cfg.ofe = None

    cfg.mprs = None
    cfg.irp = None
    cfg.pator = None
    cfg.msadre = None
    cfg.orp = None

    # ==================================================================
    # Calculate metrics relevant for the Base Target Model.
    # ==================================================================
    if model_type == "Base Target":

        cfg.mae = mean_absolute_error(
            y_actual_for_metrics_plots_u, y_preds_for_metrics_plots_u
            )
        cfg.mae_base_target_model = cfg.mae

        cfg.mse = mean_squared_error(
            y_actual_for_metrics_plots_u, y_preds_for_metrics_plots_u
            )

    # ==================================================================
    # Calculate metrics relevant for Ceiling Models.
    # ==================================================================
    elif model_type == "Ceiling":

        # Calculate Portion of Actual Targets Greater Than Ceiling (PATGTC):
        actual_targets_greater_than_ceiling_distance_list = []
        for i in range(len(y_actual_for_metrics_plots_u)):
            if y_actual_for_metrics_plots_u[i] > y_preds_for_metrics_plots_u[i]:
                actual_targets_greater_than_ceiling_distance_list.append(
                    y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]
                    )
        if (len(actual_targets_greater_than_ceiling_distance_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
            cfg.patgtc = \
                len(actual_targets_greater_than_ceiling_distance_list) \
                / len(y_actual_for_metrics_plots_u)
        else:
            cfg.patgtc = 0.0

        # Calculate Adjusted Mean Out-of-Range Proportional Distance 
        # Above Ceiling (AMORPDAC):
        actual_targets_greater_than_ceiling_distance_proportion_list = []

        for i in range(len(y_actual_for_metrics_plots_u)):
            if (y_actual_for_metrics_plots_u[i] > y_preds_for_metrics_plots_u[i]) \
                    and (y_preds_for_metrics_plots_u[i] != 0):
                actual_targets_greater_than_ceiling_distance_proportion_list.append(
                    ((y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]) \
                        / y_preds_for_metrics_plots_u[i])**2
                    )
        if (len(actual_targets_greater_than_ceiling_distance_proportion_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
            cfg.amorpdac = \
                len(actual_targets_greater_than_ceiling_distance_proportion_list)**2 \
                * statistics.mean(
                    actual_targets_greater_than_ceiling_distance_proportion_list
                    ) \
                / len(y_actual_for_metrics_plots_u)
        else:
            # If all of the actual target values exceeded the predictions...
            if cfg.patgtc == 1.0:
                cfg.amorpdac = np.inf
            else:
                cfg.amorpdac = 0.0


        # Calculate Adjusted Mean In-Range Proportional Distance 
        # Below Ceiling (AMIRPDBC):
        actual_targets_less_than_ceiling_distance_proportion_list = []
        for i in range(len(y_actual_for_metrics_plots_u)):
            if (y_actual_for_metrics_plots_u[i] <= y_preds_for_metrics_plots_u[i]) \
                    and (y_preds_for_metrics_plots_u[i] != 0):
                actual_targets_less_than_ceiling_distance_proportion_list.append(
                    (abs(y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]) \
                        / y_preds_for_metrics_plots_u[i])**2
                    )
        if (len(actual_targets_less_than_ceiling_distance_proportion_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
             cfg.amirpdbc = \
                len(actual_targets_less_than_ceiling_distance_proportion_list) \
                * ((statistics.mean(
                    actual_targets_less_than_ceiling_distance_proportion_list
                    ))**0.5) \
                / len(y_actual_for_metrics_plots_u)
        else:
            cfg.amirpdbc = np.inf

        # Calculate Overall Ceiling Error (OCE): this is the sum of 
        # AMORPDAC and AMIRPDBC for a given model. (We don’t take 
        # the product, so as not to lose information in a case where one
        # of its elements equals zero.) We want to minimize this.
        cfg.oce = cfg.amorpdac + cfg.amirpdbc

    # ==================================================================
    # Calculate metrics relevant for Floor Models.
    # ==================================================================
    elif model_type == "Floor":

        # Calculate Portion of Actual Targets Less Than Floor (PATLTF):
        actual_targets_less_than_floor_distance_list = []
        for i in range(len(y_actual_for_metrics_plots_u)):
            if y_actual_for_metrics_plots_u[i] < y_preds_for_metrics_plots_u[i]:
                actual_targets_less_than_floor_distance_list.append(
                    y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]
                    )
        if (len(actual_targets_less_than_floor_distance_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
            cfg.patltf = \
                len(actual_targets_less_than_floor_distance_list) \
                / len(y_actual_for_metrics_plots_u)
        else:
            cfg.patltf = 0.0

        # Calculate Adjusted Mean Out-of-Range Proportional Distance 
        # Below Floor (AMORPDBF):
        actual_targets_less_than_floor_distance_proportion_list = []

        # NOTE! Closer attention should be given to how metrics handle
        # correct predicted floors with a value of 0.0.
        for i in range(len(y_actual_for_metrics_plots_u)):
            if (y_actual_for_metrics_plots_u[i] < y_preds_for_metrics_plots_u[i]) \
                    and (y_preds_for_metrics_plots_u[i] != 0):
                actual_targets_less_than_floor_distance_proportion_list.append(
                    ((y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]) \
                        / y_preds_for_metrics_plots_u[i])**2
                    )
        if (len(actual_targets_less_than_floor_distance_proportion_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
            cfg.amorpdbf = \
                len(actual_targets_less_than_floor_distance_proportion_list)**2 \
                * statistics.mean(
                    actual_targets_less_than_floor_distance_proportion_list
                    ) \
                / len(y_actual_for_metrics_plots_u)
        else:
            # If all of the actual target values were less than the 
            # predictions...
            if cfg.patltf == 1.0:
                cfg.amorpdbf = np.inf
            else:
                cfg.amorpdbf = 0.0

        # Calculate Adjusted Mean In-Range Proportional Distance 
        # Above Floor (AMIRPDAF):
        actual_targets_greater_than_floor_distance_proportion_list = []
        for i in range(len(y_actual_for_metrics_plots_u)):
            if (y_actual_for_metrics_plots_u[i] < y_preds_for_metrics_plots_u[i]) \
                    and (y_preds_for_metrics_plots_u[i] != 0):
                actual_targets_greater_than_floor_distance_proportion_list.append(
                    (abs(y_actual_for_metrics_plots_u[i] - y_preds_for_metrics_plots_u[i]) \
                        / y_preds_for_metrics_plots_u[i])**2
                    )
        if (len(actual_targets_greater_than_floor_distance_proportion_list) > 0) \
                and (len(y_actual_for_metrics_plots_u) > 0):
             cfg.amirpdaf = \
                len(actual_targets_greater_than_floor_distance_proportion_list) \
                * ((statistics.mean(
                    actual_targets_greater_than_floor_distance_proportion_list
                    ))**0.5) \
                / len(y_actual_for_metrics_plots_u)
        else:
            cfg.amirpdaf = np.inf

        # Calculate Overall Floor Error (OFE): this is the sum of 
        # AMORPDBF and AMIRPDAF for a given model. (We don’t take 
        # the product, so as not to lose information in a case where one
        # of its elements equals zero.) We want to minimize this.
        cfg.ofe = cfg.amorpdbf + cfg.amirpdaf

    # ==================================================================
    # Calculate metrics relevant for Joint Range Models.
    # ==================================================================
    elif model_type == "Joint Range":

        predicted_ceilings_list \
            = cfg.models_info_dict[cfg.model_id_curr]["predicted_ceilings_list"]
        predicted_floors_list \
            = cfg.models_info_dict[cfg.model_id_curr]["predicted_floors_list"]
        predicted_targets_list = cfg.y_preds_base_target_curr
        actual_targets_list \
            = cfg.y_valid_slctd_df.iloc[:,0].tolist()

        # Calculate Mean Proportional Range Size (MPRS):
        temp_df = pd.DataFrame(
            list(zip(
                predicted_ceilings_list,
                predicted_targets_list,
                actual_targets_list,
                predicted_floors_list
                )),
            columns=[
                "predicted_ceiling",
                "predicted_target",
                "actual_target",
                "predicted_floor"
                ]
            )

        temp_df["range_size"] = abs(
            temp_df["predicted_ceiling"] - temp_df["predicted_floor"]
            )

        temp_df["proportional_range_size"] = None
        for i in range(len(temp_df)):
            if temp_df["predicted_target"].values[i] != 0:
                temp_df["proportional_range_size"].values[i] \
                    = temp_df["range_size"].values[i] \
                    / temp_df["predicted_target"].values[i]

        cfg.mprs = temp_df["proportional_range_size"].mean()

        # Calculate Inverted Range Portion (IRP):
        temp_df["inverted_range"] = 0
        for i in range(len(temp_df)):
            if temp_df["predicted_ceiling"].values[i] \
                    < temp_df["predicted_floor"].values[i]:
                temp_df["inverted_range"].values[i] = 1

        cfg.irp = temp_df["inverted_range"].sum() \
            / len(actual_targets_list)

        # Calculate Portion of Actual Targets Out of Range (PATOR):
        actual_targets_out_of_range_distance_list = []
        for i in range(len(actual_targets_list)):
            # If an actual target was above the predicted ceiling...
            if actual_targets_list[i] > predicted_ceilings_list[i]:
                actual_targets_out_of_range_distance_list.append(
                    abs(actual_targets_list[i] - predicted_ceilings_list[i])
                    )
            # If an actual target was below the predicted floor...
            if actual_targets_list[i] < predicted_floors_list[i]:
                actual_targets_out_of_range_distance_list.append(
                    abs(actual_targets_list[i] - predicted_floors_list[i])
                    )
        if (len(actual_targets_out_of_range_distance_list) > 0) \
                and (len(actual_targets_list) > 0):
            cfg.pator = \
                len(actual_targets_out_of_range_distance_list) \
                / len(actual_targets_list)
        else:
            cfg.pator = 0.0


        # Calculate Mean Summed Absolute Distances to Range Edges 
        # (MSADRE):
        actual_targets_distances_to_ceiling_floor_list = []

        for i in range(len(actual_targets_list)):

            # Calculate the absolute distance from the actual target
            # value to the predicted ceiling.
            c_dist = abs(actual_targets_list[i] - predicted_ceilings_list[i])

            # Calculate the absolute distance from the actual target
            # value to the predicted floor.
            f_dist = abs(actual_targets_list[i] - predicted_floors_list[i])

            dist_sum = (c_dist + f_dist)

            actual_targets_distances_to_ceiling_floor_list.append(dist_sum)

        if (len(actual_targets_list) > 0):
            cfg.msadre = \
                sum(actual_targets_distances_to_ceiling_floor_list) \
                / len(actual_targets_list)
        else:
            cfg.msadre = np.inf

        # Calculate Overall Range Performance (ORP) for a given model. We 
        # want to maximize this.
        cfg.orp = ((1.0 - cfg.pator)**2) / (cfg.msadre**0.5)


        # Generate a plot for this Joint Range Model.
        vis.plot_model_results_joint_range_scatter(
            temp_df,
            cfg.models_info_dict[cfg.model_id_curr]["model_title_for_plot_display"],
            cfg.models_info_dict[cfg.model_id_curr]["model_name_for_plot_filename"]
            )

    # ==================================================================
    # Add this model's metrics to the DF of models' metrics.
    # ==================================================================

    # Create a new row in cfg.models_metrics_df that includes the model's
    # metrics and other key data.
    cfg.models_metrics_df.loc[len(cfg.models_metrics_df)] = [

        # General info.
        cfg.models_info_dict[cfg.model_id_curr]["model_id"],
        model_type, # "Base Target", "Ceiling", "Floor", "Joint Range"
        cfg.models_info_dict[cfg.model_id_curr]["model_parameters_description"],

        # Relevant for the Base Target Model.
        cfg.mae,
        cfg.mse,

        # Relevant for a Ceiling Model.
        cfg.patgtc,
        cfg.amorpdac,
        cfg.amirpdbc,
        cfg.oce,

        # Relevant for a Floor Model.
        cfg.patltf,
        cfg.amorpdbf,
        cfg.amirpdaf,
        cfg.ofe,

        # Relevant for a Joint Range Model.
        cfg.mprs,
        cfg.irp,
        cfg.pator,
        cfg.msadre,
        cfg.orp,
        ]

    cfg.models_metrics_df = \
        cfg.models_metrics_df.sort_values(by="OCE", ascending=True)
    cfg.models_metrics_df = \
        cfg.models_metrics_df.sort_values(by="OFE", ascending=True)
    cfg.models_metrics_df = \
        cfg.models_metrics_df.sort_values(by="ORP", ascending=False)

    """
    # For following results in progress.
    print(tabulate(
        cfg.models_metrics_df,
        headers = "keys",
        tablefmt = "psql"
        ))
    """


def lowest_safe_ceiling_loss_func(y_true, y_pred):
    """
    This loss function is designed to help with finding the lowest number
    that comes as close as possible to the actual target value (from above),
    *without* actually being less than it.

    PARAMETERS
    ----------
    y_true
        The actual target values
    y_pred
        The predicted (target, ceiling, or floor) values
    """

    # ==================================================================
    # Loss functions that use a single formula for penalizing both 
    # in-range and out-of-range actual error.
    # ==================================================================

    # ------------------------------------------------------------------
    # Type clng_cstm_A
    # 
    # Penalty: (squared diff) ± n × (mean diff)
    # ------------------------------------------------------------------
    if cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_1":
        factor_m = 0.5
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_2":
        factor_m = 0.55
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_3":
        factor_m = 0.6
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_4":
        factor_m = 0.65
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_5":
        factor_m = 0.7
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_6":
        factor_m = 0.75
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_7":
        factor_m = 0.8
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_8":
        factor_m = 0.85
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_9":
        factor_m = 0.9
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_10":
        factor_m = 1.0
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_A_11":
        factor_m = 1.1
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    # ------------------------------------------------------------------
    # Type clng_cstm_B
    #
    # Penalty: squared diff (calculated in various ways
    # ------------------------------------------------------------------
    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_B_1":
        difference = math_ops.squared_difference(y_pred, y_true)
        loss = K.mean(difference, axis=-1)

    # ==================================================================
    # Loss functions that use different formulas for penalizing
    # in-range versus out-of-range actual error.
    # ==================================================================

    # ------------------------------------------------------------------
    # Type clng_cstm_C
    #
    # Out-of-range penalty: involving proportionality
    # ------------------------------------------------------------------
    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_C_1":
        factor_m = 50.0
        loss_pred_safely_high = K.square((y_pred - y_true) / y_pred)
        loss_pred_too_low = factor_m * K.square((y_pred - y_true) / y_pred)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    # ------------------------------------------------------------------
    # Type clng_cstm_D
    #
    # Out-of-range penalty: n × (sqr diff)
    # ------------------------------------------------------------------
    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_1":
        factor_m = 10.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_2":
        factor_m = 30.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_3":
        factor_m = 50.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_4":
        factor_m = 100.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_5":
        factor_m = 100.0
        loss_pred_safely_high = K.abs(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_D_6":
        factor_m = 200.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    # ------------------------------------------------------------------
    # Type clng_cstm_E
    #
    # Out-of-range penalty: n × (diff ^ 8)
    # ------------------------------------------------------------------
    elif cfg.lowest_safe_ceiling_loss_func_id == "clng_cstm_E_1":
        factor_m = 10.0
        loss_pred_safely_high = K.square(y_pred - y_true)
        loss_pred_too_low = factor_m * K.square(K.square(K.square(y_pred - y_true)))
        loss = K.switch((
            K.greater_equal(y_pred, y_true)), loss_pred_safely_high, loss_pred_too_low
            )

    return loss


def highest_safe_floor_loss_func(y_true, y_pred):
    """
    This loss function is designed to help with finding the highest number
    that comes as close as possible to the actual target value (from below),
    *without* actually being greater than it.

    PARAMETERS
    ----------
    y_true
        The actual target values
    y_pred
        The predicted (target, ceiling, or floor) values
    """

    # ==================================================================
    # Loss functions that use a single formula for penalizing both 
    # in-range and out-of-range actual error.
    # ==================================================================

    # ------------------------------------------------------------------
    # Type flr_cstm_A
    # 
    # Penalty: (squared diff) ± n × (mean diff)
    # ------------------------------------------------------------------
    if cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_1":
        factor_m = -0.5
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_2":
        factor_m = -0.55
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_3":
        factor_m = -0.6
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_4":
        factor_m = -0.65
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_5":
        factor_m = -0.7
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_6":
        factor_m = -0.75
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_7":
        factor_m = -0.8
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_8":
        factor_m = -0.85
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_9":
        factor_m = -0.9
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_10":
        factor_m = -1.0
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_A_11":
        factor_m = -1.1
        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1) \
            + (factor_m * K.mean(y_true - y_pred))

    # ------------------------------------------------------------------
    # Type flr_cstm_B
    #
    # Penalty: squared diff (calculated in various ways
    # ------------------------------------------------------------------
    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_B_1":
        difference = math_ops.squared_difference(y_pred, y_true)
        loss = K.mean(difference, axis=-1)

    # ==================================================================
    # Loss functions that use different formulas for penalizing
    # in-range versus out-of-range actual error.
    # ==================================================================

    # ------------------------------------------------------------------
    # Type flr_cstm_C
    #
    # Out-of-range penalty: involving proportionality
    # ------------------------------------------------------------------
    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_C_1":
        factor_m = 50.0
        loss_pred_safely_low = K.square((y_pred - y_true) / y_pred)
        loss_pred_too_high = factor_m * K.square((y_pred - y_true) / y_pred)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    # ------------------------------------------------------------------
    # Type flr_cstm_D
    #
    # Out-of-range penalty: n × (sqr diff)
    # ------------------------------------------------------------------
    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_1":
        factor_m = 10.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_2":
        factor_m = 30.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_3":
        factor_m = 50.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_4":
        factor_m = 100.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_5":
        factor_m = 100.0
        loss_pred_safely_low = K.abs(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_D_6":
        factor_m = 200.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(y_pred - y_true)
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    # ------------------------------------------------------------------
    # Type flr_cstm_E
    #
    # Out-of-range penalty: n × (diff ^ 8)
    # ------------------------------------------------------------------
    elif cfg.highest_safe_floor_loss_func_id == "flr_cstm_E_1":
        factor_m = 10.0
        loss_pred_safely_low = K.square(y_pred - y_true)
        loss_pred_too_high = factor_m * K.square(K.square(K.square(y_pred - y_true)))
        loss = K.switch((
            K.less_equal(y_pred, y_true)), loss_pred_safely_low, loss_pred_too_high
            )

    return loss


def validate_n_sd_ceiling_model(
    sd_multiplier_u
    ):
    """
    Generates a model that predicts ceiling values by adding some
    multiple of a person's historical SD to the predicted target
    value.

    PARAMETERS
    ----------
    sd_multiplier_u
        The factor by which the SD of a person's historical scores
        should be multiplied when predicting a range ceiling
    """

    model_id = "clng_SD_" + str(sd_multiplier_u)

    cfg.models_info_dict[model_id] = {
        "model_id": model_id,
        "model_type": "Ceiling",
        "model_parameters_description": "predicted target value + " + str(sd_multiplier_u) + "×SD",
        "model_title_for_plot_display": "predicted target value + " + str(sd_multiplier_u) + "×SD",
        "model_name_for_plot_filename": "clng pred target value + " + str(sd_multiplier_u) + "×SD",
        }

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = model_id

    # Calculate the predicted ceiling values. Take the predicted y values
    # for the target itself, and add N*SD for the given row.
    y_preds_target_list = cfg.y_preds_base_target_curr.tolist()
    sd_values_list = cfg.X_valid_slctd_df_preprocessed[
        cfg.feature_for_calculating_target_range
        ].tolist()

    y_preds_ceiling_list = list()
    for target_pred, sd_value in zip(y_preds_target_list, sd_values_list):
        y_preds_ceiling_list.append(target_pred + (sd_value * sd_multiplier_u))
    y_preds_ceiling_list \
        = [ceiling if ceiling > 0 else 0 for ceiling in y_preds_ceiling_list]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_ceiling_list

    # Calculate the largest calculated ±y-value associated with this 
    # model, for use in setting the y-max in Joint Range plots.
    y_max_candidates_for_plotting_list = list()
    for predicted_ceiling, predicted_target in zip(
            y_preds_ceiling_list,
            cfg.y_preds_base_target_curr,
            ):
        y_max_candidates_for_plotting_list.append(abs(predicted_ceiling - predicted_target))
        y_max_candidates_max = max(y_max_candidates_for_plotting_list)
    cfg.models_info_dict[cfg.model_id_curr]["y_max_candidate_for_joint_range_plots"] \
        = y_max_candidates_max

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_ceiling_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_ceiling_list,
        cfg.y_valid_slctd_df,
        "ceiling"
        )


def validate_n_sd_floor_model(
    sd_multiplier_u
    ):
    """
    Generates a model that predicts floor values by subtracing some
    multiple of a person's historical SD from the predicted target
    value.

    PARAMETERS
    ----------
    sd_multiplier_u
        The factor by which the SD of a person's historical scores
        should be multiplied when predicting a range floor
    """

    model_id = "flr_SD_" + str(sd_multiplier_u)

    cfg.models_info_dict[model_id] = {
        "model_id": model_id,
        "model_type": "Floor",
        "model_parameters_description": "predicted target value - " + str(sd_multiplier_u) + "×SD",
        "model_title_for_plot_display": "predicted target value - " + str(sd_multiplier_u) + "×SD",
        "model_name_for_plot_filename": "flr pred target value - " + str(sd_multiplier_u) + "×SD",
        }

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = model_id


    # Calculate the predicted floor values. Take the predicted y values
    # for the target itself, and subtract N*SD for the given row.
    # If the difference is less than 0, set the predicted floor to 0.
    y_preds_target_list = cfg.y_preds_base_target_curr.tolist()
    sd_values_list = cfg.X_valid_slctd_df_preprocessed[
        cfg.feature_for_calculating_target_range
        ].tolist()

    y_preds_floor_list = list()
    for target_pred, sd_value in zip(y_preds_target_list, sd_values_list):
        y_preds_floor_list.append(target_pred - (sd_value * sd_multiplier_u))
    y_preds_floor_list = [floor if floor > 0 else 0 for floor in y_preds_floor_list]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_floor_list

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_floor_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_floor_list,
        cfg.y_valid_slctd_df,
        "floor"
        )


def validate_mae_ceiling_model(
    ):
    """
    Generates a model that predicts ceiling values by adding the Base
    Target Model's MAE to the predicted target value.
    """

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = "clng_MAE"

    # Calculate the predicted ceiling values. Take the predicted y values
    # for the target itself, and add the base target model's MAE on
    # the training data.

    y_preds_target_list = cfg.y_preds_base_target_curr.tolist()
    y_preds_ceiling_list \
        = [pred + cfg.mae_base_target_model for pred in y_preds_target_list]
    y_preds_ceiling_list \
        = [ceiling if ceiling > 0 else 0 for ceiling in y_preds_ceiling_list]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_ceiling_list

    # Calculate the largest calculated ±y-value associated with this 
    # model, for use in setting the y-max in Joint Range plots.
    y_max_candidates_for_plotting_list = list()
    for predicted_ceiling, predicted_target in zip(
            y_preds_ceiling_list,
            cfg.y_preds_base_target_curr,
            ):
        y_max_candidates_for_plotting_list.append(abs(predicted_ceiling - predicted_target))
        y_max_candidates_max = max(y_max_candidates_for_plotting_list)
    cfg.models_info_dict[cfg.model_id_curr]["y_max_candidate_for_joint_range_plots"] \
        = y_max_candidates_max

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_ceiling_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_ceiling_list,
        cfg.y_valid_slctd_df,
        "ceiling"
        )


def validate_mae_floor_model(
    ):
    """
    Generates a model that predicts ceiling values by subtracting the 
    Base Target Model's MAE to the predicted target value.
    """

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = "flr_MAE"

    # Calculate the predicted ceiling values. Take the predicted y values
    # for the target itself, and subtract the base target model's MAE on
    # the training data.
    # If the difference is less than 0, set the predicted floor to 0.

    y_preds_target_list = cfg.y_preds_base_target_curr.tolist()
    y_preds_floor_list \
        = [pred - cfg.mae_base_target_model for pred in y_preds_target_list]
    y_preds_floor_list \
        = [floor if floor > 0 else 0 for floor in y_preds_floor_list]
    cfg.models_info_dict[cfg.model_id_curr]["y_preds_list"] = y_preds_floor_list

    # Generate metrics and plots.
    generate_metrics_for_model(
        cfg.y_valid_slctd_df,
        y_preds_floor_list,
        )
    vis.plot_model_results_floor_or_ceiling_scatter(
        y_preds_floor_list,
        cfg.y_valid_slctd_df,
        "floor"
        )


def create_joint_range_models_from_floor_ceiling_models():
    """
    Creates a joint range model by combining a particular floor model
    with a particular ceiling mode.
    """

    # ==================================================================
    # Create a Joint Range Model based on ± MAE ceilings and floors.
    # ==================================================================

    model_id = "jnt_rng_MAE"

    # Setting this value is crucial for configuring of the model
    # and the correct generation of metrics and plots.
    cfg.model_id_curr = model_id

    cfg.models_info_dict[model_id] = {
        "model_id": model_id,
        "model_type": "Joint Range",
        "model_parameters_description": "predicted target value ± Base Target Model MAE",
        "model_title_for_plot_display": "predicted target value ± Base Target Model MAE\n",
        "model_name_for_plot_filename": "predicted target value ± Base Target Model MAE",
        "predicted_floors_list": cfg.models_info_dict["flr_MAE"]["y_preds_list"],
        "predicted_ceilings_list": cfg.models_info_dict["clng_MAE"]["y_preds_list"],
        }

    generate_metrics_for_model(
        None,
        None,
        )

    # ==================================================================
    # Create Joint Range Models based on ± (n × SD) ceilings and floors.
    # ==================================================================

    for sd_multiplier in cfg.SD_MULTIPLIERS_TO_USE:

        model_id = "jnt_rng_SD_" + str(sd_multiplier)

        # Setting this value is crucial for configuring of the model
        # and the correct generation of metrics and plots.
        cfg.model_id_curr = model_id

        cfg.models_info_dict[model_id] = {
            "model_id": model_id,
            "model_type": "Joint Range",
            "model_parameters_description": "predicted target value ± " + str(sd_multiplier) + "×SD",
            "model_title_for_plot_display": "predicted target value ± " + str(sd_multiplier) + "×SD\n",
            "model_name_for_plot_filename": "predicted target value ± " + str(sd_multiplier) + "×SD",
            "predicted_floors_list": cfg.models_info_dict["flr_SD_" + str(sd_multiplier)]["y_preds_list"],
            "predicted_ceilings_list": cfg.models_info_dict["clng_SD_" + str(sd_multiplier)]["y_preds_list"],
            }

        generate_metrics_for_model(
            None,
            None,
            )

    # ==================================================================
    # Create Joint Range Models from the best-performing Ceiling and
    # Floor Models that employ custom loss functions.
    # ==================================================================

    # ------------------------------------------------------------------
    # Identify the 3 best-performing custom Ceiling Models and 3 best-
    # performing custom Floor Models.
    # ------------------------------------------------------------------

    temp_df = cfg.models_metrics_df.copy()

    # Make a DF with only the custom Ceiling Models.
    ceilings_df = temp_df[temp_df["model_type"] == "Ceiling"]
    ceilings_df = ceilings_df[ceilings_df["model_id"].str.contains("_cstm_") ]

    # The DF is already sorted with the lowest OCE scores at the top.
    # Get the ID of the best-performing models.
    model_id_custom_ceiling_best = {}
    model_id_custom_ceiling_best[0] = ceilings_df["model_id"].values[0]
    model_id_custom_ceiling_best[1] = ceilings_df["model_id"].values[1]
    model_id_custom_ceiling_best[2] = ceilings_df["model_id"].values[2]
    model_id_custom_ceiling_best[3] = ceilings_df["model_id"].values[3]
    model_id_custom_ceiling_best[4] = ceilings_df["model_id"].values[4]

    # Make a DF with only the custom Floor Models.
    floors_df = temp_df[temp_df["model_type"] == "Floor"]
    floors_df = floors_df[floors_df["model_id"].str.contains("_cstm_") ]

    # The DF is already sorted with the lowest OFE scores at the top.
    # Get the ID of the best-performing models.
    model_id_custom_floor_best = {}
    model_id_custom_floor_best[0] = floors_df["model_id"].values[0]
    model_id_custom_floor_best[1] = floors_df["model_id"].values[1]
    model_id_custom_floor_best[2] = floors_df["model_id"].values[2]
    model_id_custom_floor_best[3] = floors_df["model_id"].values[3]
    model_id_custom_floor_best[4] = floors_df["model_id"].values[4]

    # ------------------------------------------------------------------
    # Create Joint Range Models by combining the best custom Ceiling
    # Model with the best custom Floor Model, the 2nd-best Ceiling
    # with the 2nd-best Floor, etc.
    # ------------------------------------------------------------------

    for custom_ceiling_best_rank in [0, 1, 2, 3, 4]:
        for custom_floor_best_rank in [0, 1, 2, 3, 4]:

            model_id = "jnt_rng_cstm_F" + str(custom_floor_best_rank + 1) \
                + "_C" + str(custom_ceiling_best_rank + 1)

            # Setting this value is crucial for configuring of the model
            # and the correct generation of metrics and plots.
            cfg.model_id_curr = model_id

            model_descript_ceiling \
                = cfg.models_info_dict[
                    model_id_custom_ceiling_best[custom_ceiling_best_rank]
                    ]["model_parameters_description"]
            model_descript_floor \
                = cfg.models_info_dict[
                    model_id_custom_floor_best[custom_floor_best_rank]
                    ]["model_parameters_description"]
            model_filename_ceiling \
                = cfg.models_info_dict[model_id_custom_ceiling_best[
                    custom_ceiling_best_rank]
                    ]["model_name_for_plot_filename"]
            model_filename_floor \
                = cfg.models_info_dict[
                    model_id_custom_floor_best[custom_floor_best_rank]
                    ]["model_name_for_plot_filename"]

            cfg.models_info_dict[model_id] = {
                "model_id": model_id,
                "model_type": "Joint Range",
                "model_parameters_description": "C: " + model_descript_ceiling + ", F: " + model_descript_floor,
                "model_title_for_plot_display": "ANNs with ceiling: " + model_descript_ceiling + ",\nfloor: " + model_descript_floor,
                "model_name_for_plot_filename": "C " + model_filename_ceiling + ", F " + model_filename_floor,
                "predicted_floors_list": cfg.models_info_dict[
                    model_id_custom_floor_best[custom_floor_best_rank]
                    ]["y_preds_list"],
                "predicted_ceilings_list": cfg.models_info_dict[
                    model_id_custom_ceiling_best[custom_ceiling_best_rank]
                    ]["y_preds_list"],
                }

            generate_metrics_for_model(
                None,
                None,
                )

    # Save the main dict of models info (with predictions) to a pickle file.
    filename_and_path = os.path.join(
        cfg.DATASETS_DIR,
        "user_generated",
        "models_info_dict_and_y_valid_slctd_df.pickle"
        )
    with open(filename_and_path, 'wb') as file:
        pickle.dump(
            (
                cfg.models_info_dict,
                cfg.y_valid_slctd_df
                ),
            file
            )


# ██████████████████████████████████████████████████████████████████████