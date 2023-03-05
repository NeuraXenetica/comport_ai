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
This module handles the tool's FastAPI web interface.
"""

import os
import shutil

import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import other modules from this package.
import config as cfg
import cai_executor as exec
import io_file_manager as iofm
import cai_ai as ai


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")

# Set up the directory structure for input files, output files, etc.
iofm.specify_directory_structure()


@app.get('/', response_class=HTMLResponse)
def get_webpage(request: Request):
    """
    Loads the Comport_AI interface as a webpage in the user's
    web browser.
    """

    # Delete any existing plots in the plots directory
    # that might remain from previous simulations.
    for file in os.listdir(cfg.PLOTS_DIR):
        os.remove(os.path.join(cfg.PLOTS_DIR, file))

    # Display the initial webpage so that the user can provide input.
    # Pass the default values for variables to be updated by the user 
    # in the form.
    return templates.TemplateResponse(
        'cai_interface.html',
        {
            "request": request,
            "models_data_source_to_display": \
                cfg.models_data_source,
            "USER_CONFIGURABLE_MODEL_SETTING_A_to_display": \
                cfg.USER_CONFIGURABLE_MODEL_SETTING_A,
            "USER_CONFIGURABLE_MODEL_SETTING_B_to_display": \
                cfg.USER_CONFIGURABLE_MODEL_SETTING_B,
            "USER_CONFIGURABLE_MODEL_SETTING_C_to_display": \
                cfg.USER_CONFIGURABLE_MODEL_SETTING_C,
            "USER_CONFIGURABLE_MODEL_SETTING_D_to_display": \
                cfg.USER_CONFIGURABLE_MODEL_SETTING_D,
            "plots_to_display_list": [],
            "dataset_csv_for_download_url_to_display": None,
            }
        )


@app.post('/', response_class=HTMLResponse)
def post_webpage(
    request: Request,
    models_data_source_from_form: str = Form(...),
    file: UploadFile = File(...),
    USER_CONFIGURABLE_MODEL_SETTING_A_from_form: str = Form(...),
    USER_CONFIGURABLE_MODEL_SETTING_B_from_form: str = Form(...),
    USER_CONFIGURABLE_MODEL_SETTING_C_from_form: str = Form(...),
    USER_CONFIGURABLE_MODEL_SETTING_D_from_form: str = Form(...),
    ):
    """
    Sends the form data inputted by the user in the webpage to be
    processed and generated an updated webpage with visualizations
    and a link to download the CSV file with the raw model metrics.
    """

    # Reset selected variables to their factory-original state.
    cfg.plots_to_display_list = []
    cfg.dataset_csv_for_download_url = None

    try:
        with open(os.path.join(
                cfg.DATASETS_DIR,
                "user_generated",
                "uploaded_file.csv"
                ), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except:
        print("Unable to process the uploaded file!")

    # Update variables stored in config.py with the 
    # user-provided values received through the form.
    cfg.models_data_source = models_data_source_from_form
    cfg.USER_CONFIGURABLE_MODEL_SETTING_A = \
        str(USER_CONFIGURABLE_MODEL_SETTING_A_from_form)
    cfg.USER_CONFIGURABLE_MODEL_SETTING_B = \
        str(USER_CONFIGURABLE_MODEL_SETTING_B_from_form)
    cfg.USER_CONFIGURABLE_MODEL_SETTING_C = \
        int(USER_CONFIGURABLE_MODEL_SETTING_C_from_form)
    cfg.USER_CONFIGURABLE_MODEL_SETTING_D = \
        str(USER_CONFIGURABLE_MODEL_SETTING_D_from_form)

    # Load or create a cfg.pers_day_df dataset derived from a WFS file.
    print("cfg.models_data_source: ", cfg.models_data_source)
    if cfg.models_data_source == "built_in_pers_day_df":
        exec.load_existing_pers_day_df_from_disk()
    elif cfg.models_data_source \
            == "user_uploaded_wfs_behaviors_and_records":
        exec.create_fresh_pers_day_df()

    ai.handle_dataset_rows_with_null_values()

    # From the cfg.pers_day_df data, generate training, validation,
    # and test sets.
    print("COMPLETED ATTEMPT TO LOAD OR GENERATE CFG.PERS_DAY_DF")
    print("len(cfg.pers_day_df): ", len(cfg.pers_day_df))
    exec.create_train_valid_test_dfs_and_extract_features_and_targets()
    print("len(cfg.Xy_train_df): ", len(cfg.Xy_train_df))

    # Build, train, and use models.
    exec.train_and_validate_models_and_generate_plots()

    cfg.plots_to_display_list = os.listdir(cfg.PLOTS_DIR)

    return templates.TemplateResponse(
        'cai_interface.html',
        {
            "request": request,
            "models_data_source_to_display": \
                cfg.models_data_source,
            "USER_CONFIGURABLE_MODEL_SETTING_A_to_display": cfg.USER_CONFIGURABLE_MODEL_SETTING_A,
            "USER_CONFIGURABLE_MODEL_SETTING_B_to_display": cfg.USER_CONFIGURABLE_MODEL_SETTING_B,
            "USER_CONFIGURABLE_MODEL_SETTING_C_to_display": cfg.USER_CONFIGURABLE_MODEL_SETTING_C,
            "USER_CONFIGURABLE_MODEL_SETTING_D_to_display": cfg.USER_CONFIGURABLE_MODEL_SETTING_D,
            "plots_to_display_list": cfg.plots_to_display_list,
            "dataset_csv_for_download_url_to_display": \
                cfg.dataset_csv_for_download_url
            }
        )

# Run the app using uvicorn.
if __name__ == '__main__':
    uvicorn.run("cai_app:app", reload=True)


# ██████████████████████████████████████████████████████████████████████