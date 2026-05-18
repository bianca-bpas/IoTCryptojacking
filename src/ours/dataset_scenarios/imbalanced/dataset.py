"""Ours dataset module."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from ours.dataset_scenarios import scenario

@dataclass
class DatasetConfig:
    id: int
    name: str
    path: str
    is_malicious: bool
    mac_filter: Optional[str] = None

DATA_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(1, "WebOS binary malicious", "Data/malicious/WebOS_binary.csv", True, "18:56:80:17:d0:ef"),
    DatasetConfig(2, "Server binary malicious", "Data/malicious/Server_Binary.csv", True, "a4:bb:6d:ac:e1:fd"),
    DatasetConfig(3, "Raspberry Webmine robust malicious", "Data/malicious/Raspberry_Webmine_Robust.csv", True, "dc:a6:32:67:66:4b"),
    DatasetConfig(4, "Raspberry binary malicious", "Data/malicious/Raspberry_Binary.csv", True, "dc:a6:32:68:35:8a"),
    DatasetConfig(5, "Raspberry Webmine aggressive malicious", "Data/malicious/Raspberry_Webmine_Aggressive.csv", True, "dc:a6:32:67:66:4b"),
    DatasetConfig(6, "Raspberry WebminePool aggressive malicious", "Data/malicious/Raspberry_WebminePool_Aggressive.csv", True, "dc:a6:32:67:66:4b"),
    DatasetConfig(7, "Server WebminePool aggressive malicious", "Data/malicious/Server_WebminePool_Aggressive.csv", True, "a4:bb:6d:ac:e1:fd"),
    DatasetConfig(8, "Laptop download benign", "Data/benign-2/Laptop/Laptop_download_benign.csv", False),
    DatasetConfig(9, "Laptop idle benign", "Data/benign-2/Laptop/Laptop_idle_benign.csv", False),
    DatasetConfig(10, "Laptop interactive benign", "Data/benign-2/Laptop/Laptop_interactive_benign.csv", False),
    DatasetConfig(11, "Laptop video benign", "Data/benign-2/Laptop/Laptop_video_benign.csv", False),
    DatasetConfig(12, "Laptop web browsing benign", "Data/benign-2/Laptop/Laptop_webbrowsing_benign.csv", False),
    DatasetConfig(13, "Raspberry download benign", "Data/benign-2/Raspberry/Raspberry_download_benign.csv", False),
    DatasetConfig(14, "Raspberry idle benign", "Data/benign-2/Raspberry/Raspberry_idle_benign.csv", False),
    DatasetConfig(15, "Raspberry interactive benign", "Data/benign-2/Raspberry/Raspberry_interactive_benign.csv", False),
    DatasetConfig(16, "Raspberry video benign", "Data/benign-2/Raspberry/Raspberry_video_benign.csv", False),
    DatasetConfig(17, "Raspberry web browsing benign", "Data/benign-2/Raspberry/Raspberry_webbrowsing_benign.csv", False),
    DatasetConfig(18, "Server download benign", "Data/benign-2/Server/Server_download_benign.csv", False),
    DatasetConfig(19, "Server idle benign", "Data/benign-2/Server/Server_idle_benign.csv", False),
    DatasetConfig(20, "Server interactive benign", "Data/benign-2/Server/Server_interactive_benign.csv", False),
    DatasetConfig(21, "Server video benign", "Data/benign-2/Server/Server_video_benign.csv", False),
    DatasetConfig(22, "Server web browsing benign", "Data/benign-2/Server/Server_webbrowsing_benign.csv", False),
    DatasetConfig(23, "Webos video benign", "Data/benign-2/WebOS/Webos_video(live&normal)_benign.csv", False),
    DatasetConfig(32, "Server WebminePool robust malicious", "Data/malicious/Server_WebminePool_Robust.csv", True, "a4:bb:6d:ac:e1:fd"),
    DatasetConfig(33, "Raspberry WebminePool stealthy malicious", "Data/malicious/Raspberry_WebminePool_Stealthy.csv", True, "dc:a6:32:67:66:4b"),
    DatasetConfig(34, "Raspberry WebminePool robust malicious", "Data/malicious/Raspberry_WebminePool_Robust.csv", True, "dc:a6:32:68:35:8a"),
    DatasetConfig(35, "Desktop WebminePool aggressive malicious", "Data/malicious/Desktop_WebminePool_Aggressive.csv", True, "d8:3b:bf:8f:ba:ba"),
]

def load_and_filter(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(config.path)
    if config.mac_filter:
        df = df[(df['HW_dst'] == config.mac_filter) | (df['Hw_src'] == config.mac_filter)]
    df.insert(7, "Is_malicious", 1 if config.is_malicious else 0)
    return df

def get_data() -> dict[int, pd.DataFrame]:
    return {c.id: load_and_filter(c) for c in DATA_CONFIGS}

def build_df(data: dict[int, pd.DataFrame], entries: list[int | tuple[int, int]]) -> pd.DataFrame:
    return pd.concat([data[e] if isinstance(e, int) else data[e[0]].iloc[: e[1]] for e in entries])

_SCENARIO_CONFIGS = {
    "laptop": {"m": [35], "b": [8, 9, 10, 11, 12], "oversample": False},
    "raspberry": {"m": [3, 4, 5, 6, 33, 34], "b": [13, 14, 15, 16, 17], "oversample": False},
    "server": {"m": [2, 7, 32], "b": [19, 20, 21, 22], "oversample": False},
    "timely": {
        "m": [(1, 2832), (2, 4680), (3, 271), (4, 48), (5, 69), (6, 72), (7, 170), (32, 175), (33, 76), (34, 48), (35, 1300)],
        "b": [(8, 422784), (9, 44376), (10, 14784), (11, 3576), (12, 34728), (13, 269400), (14, 73), (15, 24144), (16, 7320), (17, 21240), (18, 544416), (19, 2664), (20, 27480), (21, 30888), (22, 12168), (23, 174648)],
        "oversample": False
    },
    "timely_oversampling": {
        "m": [1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35],
        "b": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "oversample": True
    },
    "webos": {"m": [1], "b": [23], "oversample": False},
}

class Scenario(scenario.Scenario):
    cols_to_remove = ["Hw_src","HW_dst"]
    
    def __init__(self, name: str) -> None:
        if name not in _SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {name}")

        config = _SCENARIO_CONFIGS[name]
        data = get_data()

        self.df_mal = build_df(data, config["m"])
        self.df_ben = build_df(data, config["b"])

        if config["oversample"]:
            self.df_mal = self.df_mal.sample(len(self.df_ben), replace=True)
