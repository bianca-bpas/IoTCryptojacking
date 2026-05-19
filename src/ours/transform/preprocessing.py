
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ours.dataset_scenarios import scenario

def make_windows(df:pd.DataFrame,window_size=10, overlap = 5):
    for i in range(0,len(df),window_size):
        window = df.iloc[max(0, i-overlap):i+window_size-overlap]
        if len(window) < window_size: continue
        yield window

def _variance(series:pd.Series):
    return series.var()
def _mean(series:pd.Series):
    return series.mean()
def _std(series:pd.Series):
    return series.std()  

def basic_feature_engineering(df:pd.DataFrame,transforms = [_variance,_mean,_std]):
    features = {}
    for col in df.columns:
        for f in transforms:
            features[col + f.__name__] = f(df[col])
    return pd.DataFrame([features])

def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = int(len(df) * 0.75)
    return df.iloc[:train_end], df.iloc[train_end:]

def z_normalize(x_train:pd.DataFrame,x_test:pd.DataFrame):
    scaler = StandardScaler()
    x_train_n = scaler.fit_transform(x_train)
    x_test_n = scaler.transform(x_test)

    return x_train_n,x_test_n


from ours.dataset_scenarios.imbalanced import dataset
ALLOWED_COLUMNS=["Time", "Length"]

def time_feature_engineering(series: pd.Series) -> dict[str, float]:
    deltas = series.diff().dropna()
    return {
        "Time_mean_interval": float(deltas.mean()) if len(deltas) > 0 else 0.0,
        "Time_var_interval": float(deltas.var()) if len(deltas) > 1 else 0.0, #type:ignore[reportArgumentType]
        "Time_std_interval": float(deltas.std()) if len(deltas) > 1 else 0.0,#type:ignore[reportArgumentType]
    }

def _classical_pipeline(df, scaler:StandardScaler|None=None):
    df = df[ALLOWED_COLUMNS]
    windows_l:list[np.ndarray] =[]
    for window in make_windows(df):
        other_feats = basic_feature_engineering(window.drop(columns=["Time"], errors="ignore"))
        time_feats = pd.DataFrame([time_feature_engineering(window["Time"])])
        combined = pd.concat([other_feats, time_feats], axis=1)
        windows_l.append(combined.to_numpy().flatten())
    
    windows_n = np.asarray(windows_l)
    if scaler is None:
        scaler = StandardScaler()
        windows_n = scaler.fit_transform(windows_n)
    else:
        windows_n = scaler.transform(windows_n)

    return windows_n,scaler


def classical_pipeline(scenario: scenario.Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns training x, training y, testing x, testing y"""
    x_ben_ta, x_ben_te = time_series_split(scenario.df_ben)
    x_mal_ta, x_mal_te = time_series_split(scenario.df_mal)
    
    normalized = []
    scaler = None
    for df in [x_ben_ta,x_ben_te,x_mal_ta,x_mal_te]:
        data,scaler = (_classical_pipeline(df, scaler))
        normalized.append(data)
    
    x_j_ta = np.concatenate([normalized[0], normalized[2]])
    y_j_ta = np.concatenate([np.zeros(len(normalized[0])), np.ones(len(normalized[2]))])

    x_j_te = np.concatenate([normalized[1], normalized[3]])
    y_j_te = np.concatenate([np.zeros(len(normalized[1])), np.ones(len(normalized[3]))])
    return x_j_ta,y_j_ta,x_j_te,y_j_te
        


if __name__ == '__main__':
    classical_pipeline(dataset.Scenario("laptop"))