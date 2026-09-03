import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path


class DataLoaderPM100:
    def __init__(
        self,
        filepath: str,
        bins_7=None,
        bins_4=None,
        split_ratio: float = 0.7,
        output_dir: str = "./datasets/pm100"
    ):
        self.filepath = filepath
        self.split_ratio = split_ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default runtime bins
        # NOTE: These bins are chosen somewhat "arbitrarily", based on a statistical analysis of the 'run_time' column of the PM100 dataset
        self.bins_7 = bins_7 or [0, 10, 200, 2000, 6000, 20000, 50000, float('inf')]
        self.labels_7 = ["Very-Short", "Short", "Medium-Short", "Medium", "Medium-Long", "Long", "Very-Long"]
        
        self.bins_4 = bins_4 or [0, 10, 300, 7200, float('inf')]
        self.labels_4 = ["Very-Short", "Short", "Medium", "Long"]
        
        self.df = None
        self.train_df = None
        self.test_df = None

    def load_data(self):
        '''
        
        Load data and perform basic preprocessing 
        
        '''
        df = pd.read_parquet(self.filepath, engine="pyarrow").copy()

        # Replace zero runtimes and remove invalid jobs
        df['run_time'] = df['run_time'].replace({0: 1})
        df = df[df['run_time'] < df['time_limit'] * 60]

        # Uniform memory requirements
        df['tot_mem_req'] = (df['mem_req'] * df['num_cores_req']) / 1024
        
        # QoS normalization
        df.qos = df.qos.replace({'normal': 1, 'qos_lowprio': 11, 'm100_qos_bprod': 4, 'm100_qos_dbg': 8})

        # Compute wait_time
        df['wait_time'] = ((df['start_time'] - df['submit_time']).dt.total_seconds()).astype(int)
        # Convert 'time_limit' in seconds to match 'run_time'
        df['time_limit'] = df['time_limit'] * 60

        # Normalize timestamps
        init_ts = int(df['submit_time'].min().timestamp())
        initial_time = datetime.utcfromtimestamp(init_ts).replace(tzinfo=pytz.UTC)
        df['submit_time_sec'] = ((df['submit_time'] - initial_time).dt.total_seconds()).astype(int)

        # --- Rename columns to target names ---
        df = df.rename(columns={
            'job_id': 'Job Number',
            'submit_time_sec': 'Submit Time',
            'wait_time': 'Wait Time',
            'run_time': 'Run Time',
            'num_nodes_req': 'Requested Number of Nodes',
            'time_limit': 'Requested Time',
            'user_id': 'User ID',
            'group_id': 'Group ID',
            'num_cores_req': 'Requested Number of CPU',
            'num_gpus_req': 'Requested Number of GPU',
            'tot_mem_req': 'Total Requested Memory',
            'qos': 'Desired QoS'
        })

        self.df = df
        return df


    def extract_additional_features(self):
        '''

        Extract additional features required for regression and classification
        
        '''
        df = self.df.copy()

        # Runtime category classification
        df["Duration (H7)"] = pd.cut(df["Run Time"], bins=self.bins_7, labels=self.labels_7, right=False)
        df["Duration (H4)"] = pd.cut(df["Run Time"], bins=self.bins_4, labels=self.labels_4, right=False)

        # Numeric durations used for historical duration features
        df['Duration (NH7)'] = df['Duration (H7)'].map({
            "Very-Short": 1, "Short": 2, "Medium-Short": 3, "Medium": 4, "Medium-Long": 5, "Long": 6, "Very-Long": 7
        }).astype(int)

        df['Duration (NH4)'] = df['Duration (H4)'].map({
            "Very-Short": 1, "Short": 2, "Medium": 3, "Long": 4
        }).astype(int)
        
        df.sort_values(by=["User ID", "Submit Time"], inplace=True)
        
        # HISTORICAL RUNTIME FEATURES
        # Prev Run Time 1 -> The running time of the last job of the same user, or 0 if such a job does not exist
        # Prev Run Time 2 -> The running time of the second-to-last job of the same user, or 0 if N/A
        # Prev Run Time 3 -> The running time of the third-to-last job of the same user, or 0 if N/A
        for n in [1, 2, 3]:
            df[f"Prev Run Time {n}"] = df.groupby("User ID")["Run Time"].shift(n).fillna(0)
        # Avg Run Time 2 -> The average running time of the two last historically recorded jobs of the same user
        df["Avg Run Time 2"] = df[["Prev Run Time 1", "Prev Run Time 2"]].mean(axis=1)
        # Avg Run Time 3 -> The average running time of the three last historically recorded jobs of the same user
        df["Avg Run Time 3"] = df[["Prev Run Time 1", "Prev Run Time 2", "Prev Run Time 3"]].mean(axis=1)
        # Avg Run Time All -> The average running time of all historically recorded jobs of the same user
        df["Avg Run Time All"] = df.groupby("User ID")["Run Time"].transform(lambda x: x.expanding().mean()).fillna(0)

        # HISTORICAL DURATION FEATURES (NH7 & NH4)
        # Similar to historical runtime features, but use (numeric) duration classes instead of runtime
        for dur_col in ["(NH7)", "(NH4)"]:
            colname = f"Duration {dur_col}"
            for n in [1, 2, 3]:
                df[f"Prev {colname} {n}"] = df.groupby("User ID")[colname].shift(n).fillna(0)
            df[f"Avg {colname} 2"] = df[[f"Prev {colname} 1", f"Prev {colname} 2"]].mean(axis=1)
            df[f"Avg {colname} 3"] = df[[f"Prev {colname} 1", f"Prev {colname} 2", f"Prev {colname} 3"]].mean(axis=1)
            df[f"Avg {colname} All"] = df.groupby("User ID")[colname].transform(lambda x: x.expanding().mean()).fillna(0)

        # RESOURCE-BASED FEATURES
        for n in [1, 2, 3]:
            df[f"Prev Requested Nodes {n}"] = df.groupby("User ID")["Requested Number of Nodes"].shift(n).fillna(0)
        # Avg Run Time 2 -> The average running time of the two last historically recorded jobs of the same user
        df["Avg Requested Nodes 2"] = df[["Prev Requested Nodes 1", "Prev Requested Nodes 2"]].mean(axis=1)
        # Avg Run Time 3 -> The average running time of the three last historically recorded jobs of the same user
        df["Avg Requested Nodes 3"] = df[["Prev Requested Nodes 1", "Prev Requested Nodes 2", "Prev Requested Nodes 3"]].mean(axis=1)
        # Avg Run Time All -> The average running time of all historically recorded jobs of the same user
        df["Avg Requested Nodes All"] = df.groupby("User ID")["Requested Number of Nodes"].transform(lambda x: x.expanding().mean()).fillna(0)

        self.df = df
        return df


    def split_train_test(self):
        '''
        
        Time consecutive split of the dataset into train_dataset and test_dataset (default split_ratio = 70-30) 
        
        '''
        df = self.df.sort_values(by="Submit Time")
        split_index = int(len(df) * self.split_ratio)
        self.train_df = df.iloc[:split_index]
        self.test_df = df.iloc[split_index:]
        return self.train_df, self.test_df


    def save_parquet(self, filename_prefix="dataset"):
        '''

        Save the full dataset, the train dataset and the test dataset in parquet files, keeping only the columns required for the prediction models

        '''
        useful_columns = [
            'Job Number', 'User ID', 'Requested Number of Nodes', 'Requested Number of CPU', 'Requested Number of GPU', 
            'Total Requested Memory', 'Desired QoS', 'Requested Time', 'Run Time', 'Duration (H4)', 'Duration (H7)',
            'Prev Run Time 1', 'Prev Run Time 2', 'Prev Run Time 3', 'Avg Run Time 2', 'Avg Run Time 3', 'Avg Run Time All', 
            'Prev Duration (NH4) 1', 'Prev Duration (NH4) 2', 'Prev Duration (NH4) 3', 'Avg Duration (NH4) 2', 'Avg Duration (NH4) 3', 'Avg Duration (NH4) All', 
            'Prev Duration (NH7) 1', 'Prev Duration (NH7) 2', 'Prev Duration (NH7) 3', 'Avg Duration (NH7) 2', 'Avg Duration (NH7) 3', 'Avg Duration (NH7) All', 
            'Prev Requested Nodes 1', 'Prev Requested Nodes 2', 'Prev Requested Nodes 3', 'Avg Requested Nodes 2', 'Avg Requested Nodes 3', 'Avg Requested Nodes All', 
            'Submit Time'
        ]

        df_full = self.df[[c for c in useful_columns if c in self.df.columns]]
        df_train = self.train_df[[c for c in useful_columns if c in self.train_df.columns]]
        df_test = self.test_df[[c for c in useful_columns if c in self.test_df.columns]]

        df_full.to_parquet(self.output_dir / f"{filename_prefix}_full.parquet", engine="pyarrow")
        df_train.to_parquet(self.output_dir / f"{filename_prefix}_train.parquet", engine="pyarrow")
        df_test.to_parquet(self.output_dir / f"{filename_prefix}_test.parquet", engine="pyarrow")

        print(f"Files saved in: {self.output_dir.resolve()}")



class DataLoaderFDATA:
    def __init__(
        self,
        filepath: str,
        bins_7=None,
        bins_4=None,
        split_ratio: float = 0.8,
        output_dir: str = "./datasets/fdata"
    ):
        self.filepath = filepath
        self.split_ratio = split_ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default runtime bins
        # NOTE: These bins are chosen somewhat "arbitrarily", based on a statistical analysis of the 'run_time' column of the PM100 dataset
        self.bins_7 = bins_7 or [0, 100, 500, 3000, 10000, 30000, 80000, float('inf')]
        self.labels_7 = ["Very-Short", "Short", "Medium-Short", "Medium", "Medium-Long", "Long", "Very-Long"]
        
        self.bins_4 = bins_4 or [0, 300, 3000, 20000, float('inf')]
        self.labels_4 = ["Very-Short", "Short", "Medium", "Long"]
        
        self.df = None
        self.train_df = None
        self.test_df = None

    
    def load_data(self):
        '''
        
        Load data and perform basic preprocessing 
        
        '''
        df = pd.read_parquet(self.filepath, engine="pyarrow").copy()

        # Convert the job id and user id from string to integer (removing the prefix)
        df['jid'] = df['jid'].str.split('_').str[1].astype(int)
        df['usr'] = df['usr'].str.split('_').str[1].astype(int)
 
        # Replace zero runtimes and remove invalid jobs
        df['duration'] = df['duration'].replace({0: 1})
        df = df[df['duration'] < df['elpl']]

        # Ensure that sdt and adt are datetime
        df["sdt"] = pd.to_datetime(df["sdt"])
        df["adt"] = pd.to_datetime(df["adt"])
        
        # Compute wait_time
        df['wait_time'] = ((df['sdt'] - df['adt']).dt.total_seconds()).astype(int)

        # Normalize timestamps
        init_ts = int(df['adt'].min().timestamp())
        initial_time = datetime.utcfromtimestamp(init_ts).replace(tzinfo=pytz.UTC)
        df['submit_time_sec'] = ((df['adt'] - initial_time).dt.total_seconds()).astype(int)

        # Add default values for Group ID and Requested Number of GPU which are not available
        df['group'] = 0
        df['gnumr'] = -1 
        
        # --- Rename columns to target names ---
        df = df.rename(columns={
            'jid': 'Job Number',
            'submit_time_sec': 'Submit Time',
            'wait_time': 'Wait Time',
            'duration': 'Run Time',
            'nnumr': 'Requested Number of Nodes',
            'elpl': 'Requested Time',
            'usr': 'User ID',
            'group': 'Group ID',
            'cnumr': 'Requested Number of CPU',
            'gnumr': 'Requested Number of GPU',
            'mszl': 'Total Requested Memory',
            'pri': 'Desired QoS',
            'embedding' : 'Embedding' 
        })

        self.df = df
        return df


    def load_preprocessed_data(self):
        df = pd.read_parquet(self.filepath, engine="pyarrow").copy()

        self.df = df
        return df
        
    def extract_additional_features(self):
        '''

        Extract additional features required for regression and classification
        
        '''
        df = self.df.copy()

        # Runtime category classification
        df["Duration (H7)"] = pd.cut(df["Run Time"], bins=self.bins_7, labels=self.labels_7, right=False)
        df["Duration (H4)"] = pd.cut(df["Run Time"], bins=self.bins_4, labels=self.labels_4, right=False)

        # Numeric durations used for historical duration features
        df['Duration (NH7)'] = df['Duration (H7)'].map({
            "Very-Short": 1, "Short": 2, "Medium-Short": 3, "Medium": 4, "Medium-Long": 5, "Long": 6, "Very-Long": 7
        }).astype(int)

        df['Duration (NH4)'] = df['Duration (H4)'].map({
            "Very-Short": 1, "Short": 2, "Medium": 3, "Long": 4
        }).astype(int)
        
        df.sort_values(by=["User ID", "Submit Time"], inplace=True)
        
        # HISTORICAL RUNTIME FEATURES
        # Prev Run Time 1 -> The running time of the last job of the same user, or 0 if such a job does not exist
        # Prev Run Time 2 -> The running time of the second-to-last job of the same user, or 0 if N/A
        # Prev Run Time 3 -> The running time of the third-to-last job of the same user, or 0 if N/A
        for n in [1, 2, 3]:
            df[f"Prev Run Time {n}"] = df.groupby("User ID")["Run Time"].shift(n).fillna(0)
        # Avg Run Time 2 -> The average running time of the two last historically recorded jobs of the same user
        df["Avg Run Time 2"] = df[["Prev Run Time 1", "Prev Run Time 2"]].mean(axis=1)
        # Avg Run Time 3 -> The average running time of the three last historically recorded jobs of the same user
        df["Avg Run Time 3"] = df[["Prev Run Time 1", "Prev Run Time 2", "Prev Run Time 3"]].mean(axis=1)
        # Avg Run Time All -> The average running time of all historically recorded jobs of the same user
        df["Avg Run Time All"] = df.groupby("User ID")["Run Time"].transform(lambda x: x.expanding().mean()).fillna(0)

        # HISTORICAL DURATION FEATURES (NH7 & NH4)
        # Similar to historical runtime features, but use (numeric) duration classes instead of runtime
        for dur_col in ["(NH7)", "(NH4)"]:
            colname = f"Duration {dur_col}"
            for n in [1, 2, 3]:
                df[f"Prev {colname} {n}"] = df.groupby("User ID")[colname].shift(n).fillna(0)
            df[f"Avg {colname} 2"] = df[[f"Prev {colname} 1", f"Prev {colname} 2"]].mean(axis=1)
            df[f"Avg {colname} 3"] = df[[f"Prev {colname} 1", f"Prev {colname} 2", f"Prev {colname} 3"]].mean(axis=1)
            df[f"Avg {colname} All"] = df.groupby("User ID")[colname].transform(lambda x: x.expanding().mean()).fillna(0)

        # RESOURCE-BASED FEATURES
                # RESOURCE-BASED FEATURES
        for n in [1, 2, 3]:
            df[f"Prev Requested Nodes {n}"] = df.groupby("User ID")["Requested Number of Nodes"].shift(n).fillna(0)
        # Avg Run Time 2 -> The average running time of the two last historically recorded jobs of the same user
        df["Avg Requested Nodes 2"] = df[["Prev Requested Nodes 1", "Prev Requested Nodes 2"]].mean(axis=1)
        # Avg Run Time 3 -> The average running time of the three last historically recorded jobs of the same user
        df["Avg Requested Nodes 3"] = df[["Prev Requested Nodes 1", "Prev Requested Nodes 2", "Prev Requested Nodes 3"]].mean(axis=1)
        # Avg Run Time All -> The average running time of all historically recorded jobs of the same user
        df["Avg Requested Nodes All"] = df.groupby("User ID")["Requested Number of Nodes"].transform(lambda x: x.expanding().mean()).fillna(0)

        self.df = df
        return df


    def split_train_test(self):
        '''
        
        Time consecutive split of the dataset into train_dataset and test_dataset (default split_ratio = 80-20) 
        
        '''
        df = self.df.sort_values(by="Submit Time")
        split_index = int(len(df) * self.split_ratio)
        self.train_df = df.iloc[:split_index]
        self.test_df = df.iloc[split_index:]
        return self.train_df, self.test_df


    def save_parquet(self, filename_prefix="dataset"):
        '''

        Save the full dataset, the train dataset and the test dataset in parquet files, keeping only the columns required for the prediction models

        '''
        useful_columns = [
            'Job Number', 'User ID', 'Requested Number of Nodes', 'Requested Number of CPU', 'Requested Number of GPU', 
            'Total Requested Memory', 'Desired QoS', 'Requested Time', 'Embedding', 'Run Time', 'Duration (H4)', 'Duration (H7)',
            'Prev Run Time 1', 'Prev Run Time 2', 'Prev Run Time 3', 'Avg Run Time 2', 'Avg Run Time 3', 'Avg Run Time All', 
            'Prev Duration (NH4) 1', 'Prev Duration (NH4) 2', 'Prev Duration (NH4) 3', 'Avg Duration (NH4) 2', 'Avg Duration (NH4) 3', 'Avg Duration (NH4) All', 
            'Prev Duration (NH7) 1', 'Prev Duration (NH7) 2', 'Prev Duration (NH7) 3', 'Avg Duration (NH7) 2', 'Avg Duration (NH7) 3', 'Avg Duration (NH7) All', 
            'Prev Requested Nodes 1', 'Prev Requested Nodes 2', 'Prev Requested Nodes 3', 'Avg Requested Nodes 2', 'Avg Requested Nodes 3', 'Avg Requested Nodes All', 
            'Submit Time'
        ]

        df_full = self.df[[c for c in useful_columns if c in self.df.columns]]
        df_train = self.train_df[[c for c in useful_columns if c in self.train_df.columns]]
        df_test = self.test_df[[c for c in useful_columns if c in self.test_df.columns]]

        df_full.to_parquet(self.output_dir / f"{filename_prefix}_full.parquet", engine="pyarrow")
        df_train.to_parquet(self.output_dir / f"{filename_prefix}_train.parquet", engine="pyarrow")
        df_test.to_parquet(self.output_dir / f"{filename_prefix}_test.parquet", engine="pyarrow")

        print(f"Files saved in: {self.output_dir.resolve()}")