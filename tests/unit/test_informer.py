import pandas as pd
import pytest
from mmf_sa.models import ModelRegistry
from omegaconf import OmegaConf
from .fixtures import m4_df

@pytest.fixture
def base_config():
    return OmegaConf.create({
        "date_col": "ds",
        "target": "y",
        "group_id": "unique_id",
        "freq": "D",
        "prediction_length": 14,
        "metric": "smape",
        "active_models": ["NeuralForecastInformer"],
        "informer_params": {
            "input_size": 14,
            "n_epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-3
        }
    })

def test_neuralforecast_informer(base_config, m4_df):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("NeuralForecastInformer")
    
    # Use a single time series for testing
    _df = m4_df[m4_df.unique_id == "D8"].copy()
    
    # Fit the model
    model.fit(_df)
    
    # Make predictions
    future_df = model.make_future_dataframe(_df, periods=14)
    predictions = model.predict(_df, future_df)
    
    # Basic assertions
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 14
    assert "ds" in predictions.columns
    assert "y" in predictions.columns
    
    # Test backtesting
    backtest_results = model.backtest(
        _df,
        start=_df.ds.max() - pd.Timedelta(days=28),
        stride=14
    )
    
    assert isinstance(backtest_results, pd.DataFrame)
    assert len(backtest_results) > 0
    assert "smape" in backtest_results.columns

    print("Backtest results:")
    print(backtest_results)
