# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
model = dbutils.widgets.get("model")

# COMMAND ----------

from forecasting_sa import run_forecast
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_output=f"{catalog}.{db}.monthly_scoring_output",
    evaluation_output=f"{catalog}.{db}.monthly_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="date",
    target="y",
    freq="M",
    prediction_length=3,
    backtest_months=12,
    stride=1,
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.monthly_ensemble_output",
    active_models=[model],
    experiment_path=f"/Shared/mmf_experiment_monthly",
    use_case_name="mmf",
    accelerator="gpu",
)