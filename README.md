# Comport_AI
Comport_AI is a free open-source HR predictive analytics tool that uses advanced machine learning to forecast the likely range of a worker’s future job performance.

Rather than mechanistically deriving the predicted ceiling and floor of a worker’s future performance from a single predicted target value using calculations based on MAE or SD, Comport_AI treats the *likely ceiling* and *likely floor* of a worker’s performance during a future timeframe as independent entities, which are modelled by artificial neural networks whose custom loss functions enable them to formulate prediction intervals that are as small as possible, while being *just large enough* to contain a worker’s actual future performance value, in the vast majority of cases. This allows more precise, nuanced, and useful forecasting of workers’ future job performance.

Comport_AI utilizes TensorFlow, Keras, scikit-learn, FastAPI, Uvicorn, Jinja2, NumPy, Pandas, and Matplotlib. It’s developed by Matthew E. Gladden (with support from Cognitive Firewall LLC and NeuraXenetica LLC) and is made available for use under GNU General Public License Version 3. Please see https://www.gnu.org/licenses/gpl-3.0.html.

___
## Steps in the range-modelling process
The process used by Comport_AI for generating optimized models for the likely range of workers’ future job performance involves the following steps.

### Launching the web app
There are various ways of running the web app, depending on how Python has been configured on one’s system. For example, presuming that one has successfully installed the Comport_AI package (e.g., through `python -m pip install comport_ai`) and its dependencies, from within Windows PowerShell, one can open the Comport_AI folder that contains the module files listed above and type: `uvicorn cai_app:app` to launch the web app, which can then be accessed through a web browser at `http://127.0.0.1:8000/`.
### Configuring the performance-range modelling process
It’s possible for a user to configure a number of the simulation’s parameters from within the web app’s interface.
**Selecting the data source.** The raw data that will be used for training models and predicting the range of workers’ future performance takes the form of a synthetic dataset of factory workers’ daily behaviors generated by the open-source Synaptans WorkforceSim platform. Comport_AI’s web interface allows a user to either upload a WorkforceSim file (of the type “wfs_behaviors_and_records_[*].csv”) or choose from one of three WorkforceSim datasets built into Comport_AI: a “Small” dataset (with roughly 20,000 observations), “Medium” dataset (≈119,000 observations), or “Large” dataset (≈412,000 observations). Building models based on the Large (or even the Medium) dataset may take a very long time or exceed the memory available on many computers; it’s thus recommended that one begin with the Small dataset to test the app’s functioning before attempting to use a larger dataset.
**Selecting the modelling settings.** A user can choose to predict either the mean value of the daily efficacy scores that will be registered for a worker in a given future time period or the number of “teamwork” behaviors that will be recorded for the worker during that period. The available time periods are the next 7 days or the next 30 days. (The modelling of workers’ registered daily efficacy scores for the next 30 days is recommended as an initial experiment, as it involves the richest dynamics and makes it easiest to visually compare different approaches to the modelling of workers’ performance ranges.) A user can also specify a random seed, in order to facilitate reproducibility of results.
### Training the performance-range models
When a user clicks on the green button to “train the performance-range models and compare their results,” the package begins working to carry out the following steps.
**Producing the Base Target Model.** Comport_AI uses a Random Forest Regressor to create its “Base Target Model,” which is trained to produce a single predicted target value (in the form of a float) as the expected future performance figure (either mean daily efficacy or the number of teamwork behaviors) for each worker in the dataset.
**Producing individual Ceiling Models.** The package then produces a number of “Ceiling Models.” The goal of a Ceiling Model is to establish the upper limit of the range of a worker’s likely future performance during the selected timeframe. An ideal predicted ceiling value is as low as possible, while still being just high enough that a worker’s actual future target value doesn’t exceed it. Ceiling Models are produced using three general methods: (1) by simply adding a model’s overall MAE to the predicted target value for a given worker; (2) by adding some multiple of the SD of a worker’s historical performance values to the predicted target value for a given worker; and (3) by constructing ANNs with custom loss functions that generate a predicted ceiling value for each worker, independently of the predicted target value. At this stage, the effectiveness of individual Ceiling Models is preliminarily assessed using the following metrics:
- **Portion of Actual Targets Greater Than Ceiling (PATGTC).** This is the number of cases in which a worker’s actual future performance value was higher than the predicted ceiling, as a share of the total number of workers. If a Ceiling Model were perfectly effective, its PATGTC would be 0.0.
- **Adjusted Mean Out-of-Range Proportional Distance Above Ceiling (AMORPDAC).** This complex metric takes into account the distance by which actual target values exceeded the predicted ceiling value, in those cases when they exceeded it. It more heavily penalizes larger distances and a larger number of cases of actual values exceeding their predicted ceiling. The higher the number, the less effective a Ceiling Model is (i.e., because many actual target values are exceeding their predicted ceiling by a large amount).
- **Adjusted Mean In-Range Proportional Distance Below Ceiling (AMIRPDBC).** This complex metric takes into account the distance by which actual target values were below the predicted ceiling value, in those cases when they were less than it. The larger the number, the less effective a Ceiling Model is (i.e., because it’s generating ceiling predictions that are unnecessarily high in value).
- **Overall Ceiling Error (OCE).** This is the sum of AMORPDAC and AMIRPDBC for a given Ceiling Model; it offers an overall measure of a Ceiling Model’s effectiveness. In the modelling of performance ranges, our goal is to minimize this number.
**Producing individual Floor Models.** Similarly, the package produces a number of “Floor Models” that seek to predict the lower limit of the range of a worker’s likely future performance during the selected timeframe. Floor Models are also produced using three methods based on (1) a model’s overall MAE; (2) the SD of a worker’s historical performance values; and (3) ANNs with custom loss functions. At this stage, the effectiveness of individual Floor Models is preliminarily assessed using the following metrics:
- **Portion of Actual Targets Less Than Floor (PATLTF).** This is the number of cases in which a worker’s actual future performance value was lower than the predicted floor, as a share of the total number of workers. If a Floor Model were perfectly effective, its PATLTF would be 0.0.
- **Adjusted Mean Out-of-Range Proportional Distance Below Floor (AMORPDBF).** This metric takes into account the distance by which actual target values fell below the predicted ceiling value, in those cases when they were less than it. It more heavily penalizes larger distances and a larger number of cases of actual values that fall below their predicted floor. The higher the number, the less effective a Floor Model is (i.e., because many actual target values are less than their predicted Floor by a large amount).
- **Adjusted Mean In-Range Proportional Distance Above Floor (AMIRPDAF).** This metric takes into account the distance by which actual target values were above the predicted floor value, in those cases when they exceeded it. The larger the number, the less effective a Floor Model is (i.e., because it’s generating floor predictions that are unnecessarily low in value).
- **Overall Floor Error (OFE).** This is the sum of AMORPDBF and AMIRPDAF for a given Floor Model; it offers an overall measure of a Floor Model’s effectiveness. In the modelling of performance ranges, our goal is to minimize this number.
**Producing Joint Range Models.** Comport_AI next combines individual Ceiling Models with individual Floor Models to produce composite “Joint Range Models” that establish the whole prediction interval for a given worker by joining a predicted ceiling value with a predicted floor value. Joint Range Models are produced using three general methods: (1) by joining the MAE-based Ceiling Model with the MAE-based Floor model; (2) by joining each of the SD-based Ceiling Models with its corresponding SD-based Floor Model; and (3) by taking the five ANN-based Ceiling Models with the lowest OCE scores and the five ANN-based Floor Models with the lowest OFE scores and creating Joint Range Models based on every possible combination of these. While the metrics for Ceiling and Floor Models previously described were useful as interim measures for narrowing the selection of ANN-based Ceiling and Floor Models to combine, those metrics are no longer relevant, once the Joint Range Models have been created. To assess the effectiveness of the Joint Range Models, we employ the metrics:
- **Mean Proportional Range Size (MPRS).** This is the distance between a worker’s predicted ceiling and floor values, in proportion to the worker’s predicted target value, averaged across all cases. An ideally functioning Joint Range Model would have an MPRS of 0.0 (i.e., the predicted ceiling and floor values would be the same as the predicted target value).
- **Inverted Range Portion (IRP).** This is the share of cases for which a worker’s predicted ceiling value is less than his or her predicted floor value. An ideally functioning Joint Range Model would have an IRP of 0.0.
- **Portion of Actual Targets Out of Range (PATOR).** This is the share of cases in which an actual target value was either greater than the predicted ceiling or less than the predicted floor. An ideally functioning Joint Range Model would have a PATOR of 0.0.
- **Mean Summed Absolute Distances to Range Edges (MSADRE).** This is sum of the absolute distance of an actual target value from its predicted ceiling value and its absolute distance from its predicted floor value, averaged across all cases.
- **Overall Range Performance (ORP).** For a given Joint Range Model, this complex metric equals `(1 – PATOR)² ÷ √(MSADRE)`. It more significantly penalizes models that have a greater share of actual target values falling outside their predicted range, while only relatively weakly penalizing models that yield larger predicted ranges. Unlike in the cases of CCE and CFE, our goal in the modelling of performance ranges is the maximize the value of ORP.

___
## Evaluating the results
Once the process of producing all of the Joint Range Models has been completed, Comport_AI will display plots visualizing the performance of the Base Target Model and Ceiling, Floor, and Joint Range Models. Each plot includes data on the given model’s metrics; it’s also possible to use the green button that will appear at the bottom of the webpage to download a CSV file with all of the model metrics.
There may be some scenarios in HR predictive analytics in which conventional Joint Range Models based on MAE or SD will yield the best Overall Range Performance (and with simpler calculations). However, there are other cases in which independently modelling ceilings and floors using ANNs with custom loss functions can generate more precise, nuanced, and useful forecasts of workers’ future job performance that may justify the additional computational resources required. By creating models using different configuration settings in Comport_AI and studying the generated visualizations and CSV files, it becomes possible to get a better sense of the situations in which advanced performance-range modelling may add business value for an organization seeking to improve its HR predictive analytics.

___
## Development
Comport_AI is free open-source software developed by Matthew E. Gladden, with support from Cognitive Firewall LLC and NeuraXenetica LLC. It is a complement to (and builds on) the open-source Synaptans WorkforceSim platform.

Compart_AI code and documentation ©2021-2023 NeuraXenetica LLC
