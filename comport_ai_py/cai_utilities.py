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
This module includes a number of general-purpose utility functions.
"""

import datetime

# Import other modules from this package.
import config as cfg


def sort_df_by_given_field_descending(
    df_u,
    col_name_u
    ):
    """
    Sorts a DataFrame by a given column (descending).

    PARAMETERS
    ----------
    df_u
        The DataFrame to be sorted
    col_name_u
        The column by which to sort
    """

    df_sorted = df_u.copy()
    df_sorted = df_sorted.sort_values(by=col_name_u, ascending=False)
    return df_sorted


def return_df_with_selected_cols_from_df(
    input_df_u,
    cols_to_keep_u
    ):
    """
    Returns a DataFrame with selected columns from an inputted 
    DataFrame.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame from which columns will be selected
    cols_to_keep_u : list
        The columns of the DataFrame that should be returned
    """

    new_df = input_df_u[cols_to_keep_u]
    return new_df


def return_df_with_selected_cols_deleted(
    input_df_u,
    cols_to_delete_u
    ):
    """
    Returns a DataFrame with selected columns deleted.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame from which columns should be deleted
    cols_to_delete_u : list
        The columns to delete
    """

    new_df = input_df_u.drop(columns=cols_to_delete_u)
    return new_df


def return_df_with_rows_filtered_to_one_val_in_col(
    input_df_u,
    col_by_which_to_filter_rows_u,
    val_to_seek_in_col
    ):
    """
    Returns a DataFrame with only those rows possessing a particular
    specified value in a particular column.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame to be filtered
    col_by_which_to_filter_rows_u
        The column by which to filter rows
    val_to_seek_in_col
        The value to seek in the column (i.e., the restrictor)
    """

    new_df = (input_df_u[input_df_u[col_by_which_to_filter_rows_u] ==
        val_to_seek_in_col])
    return new_df


def return_df_with_col_one_hot_encoded(
    input_df_u,
    col_to_one_hot_encode_u
    ):
    """
    Returns a DataFrame with new columns added that one-hot encode
    a specified already existing column.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame to one-hot encode
    col_to_one_hot_encode_u
        The column to one-hot encode
    """

    def ohe_apply(
        value_in_df,
        value_to_ohe
        ):
        """
        Returns a one-hot-encoded value of 1 or 0.

        PARAMETERS
        ----------
        value_in_df
            The term that is actually present in a DataFrame cell
        value_to_ohe : str
            The term which, if present, should produce a 1
        """

        if value_in_df == value_to_ohe:
            return int(1)
        else:
            return 0

    new_df = input_df_u.copy()

    # Create the blank new one-hot-encoding columns and populate
    # their values.
    for item in new_df[col_to_one_hot_encode_u].unique().tolist():
        col_name = str(item) + " (" + str(col_to_one_hot_encode_u) + ")"
        new_df[col_name] = None

        new_df[col_name] = new_df[col_to_one_hot_encode_u].apply(
            ohe_apply,
            value_to_ohe=item
            )

    return new_df


def return_df_with_rows_deleted_that_containing_na_in_col(
    input_df_u, col_u):
    """
    Returns a DataFrame in which rows have been deleted that include
    an Na value in a specified column.

    PARAMETERS
    ----------
    input_df_u
        The DataFrame with rows to be deleted
    col_u
        The column in which an Na value will cause row deletion
    """

    input_df_u = input_df_u[input_df_u[col_u].notna()]
    return input_df_u


def begin_tracking_elapsed_processing_time():
    """
    Stores the real-world datetime at which processing of the 
    simulation began.
    """

    cfg.sim_processing_start_datetime = datetime.datetime.now()


def return_elapsed_processing_time():
    """
    Returns the total elapsed time for which processing of some part
    of the simulation has been running.
    """

    elapsed_datetime_timedelta = (datetime.datetime.now()
        - cfg.sim_processing_start_datetime)
    elapsed_datetime_timedelta_displayable_str = \
        str(elapsed_datetime_timedelta)
    return elapsed_datetime_timedelta_displayable_str


# ██████████████████████████████████████████████████████████████████████