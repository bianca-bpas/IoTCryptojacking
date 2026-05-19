
import pandas as pd

class Scenario:
    df_mal: pd.DataFrame
    df_ben: pd.DataFrame
    cols_to_remove: list[str]

    def __init__(self, name:str):
        ...