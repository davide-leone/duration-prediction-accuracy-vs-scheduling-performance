import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

class PrintSummary:

    def show_prediction_summary(self, df):
        '''

        Calculates and shows the prediction summary
        
        '''
        df['Prediction_Type'] = df.apply(
            lambda row: 'Exact' if row['Predicted_Runtime'] == row['Actual_Runtime'] 
                        else 'Overestimate' if row['Predicted_Runtime'] > row['Actual_Runtime'] 
                        else 'Underestimate', axis=1)
        
        # Calculate percentages and average error for each category
        summary_table = df.groupby('Prediction_Type').agg(
            Count=('Job_ID', 'count'),
            Percentage=('Job_ID', lambda x: (len(x) / len(df)) * 100),
            Average_Error=('Prediction_Error', 'mean')
        ).reset_index()
        
        return pd.DataFrame(summary_table)

 
    def show_runtime_frequency(self, df):
        ''' 
        
        Count unique predicted runtime and get their frequency
        
        '''
        unique_requested_runtimes = df['Requested_Runtime'].nunique()
        print(f"Number of unique requested runtimes: {unique_requested_runtimes}")
        requested_runtime_frequency = df['Requested_Runtime'].value_counts().reset_index()
        requested_runtime_frequency.columns = ['Requested_Runtime', 'Frequency']
    
        unique_actual_runtimes = df['Actual_Runtime'].nunique()
        print(f"Number of unique actual runtimes: {unique_actual_runtimes}")
        actual_runtime_frequency = df['Actual_Runtime'].value_counts().reset_index()
        actual_runtime_frequency.columns = ['Actual_Runtime', 'Frequency']
        
        unique_predicted_runtimes = df['Predicted_Runtime'].nunique()
        print(f"Number of unique predicted runtimes: {unique_predicted_runtimes}")
        predicted_runtime_frequency = df['Predicted_Runtime'].value_counts().reset_index()
        predicted_runtime_frequency.columns = ['Predicted_Runtime', 'Frequency']
        
        # Return the frequency tables
        return (requested_runtime_frequency, actual_runtime_frequency, predicted_runtime_frequency)


    def print_metrics(self, df):
        mae = mean_absolute_error(df['Actual_Runtime'], df['Predicted_Runtime'])
        rmse = np.sqrt(mean_squared_error(df['Actual_Runtime'], df['Predicted_Runtime']))
        
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")


class ShowPlots:

    def show_runtime_histogram(self, col1, col2, label1, label2, max_time_limit=86400, path=''):
        plt.figure(figsize=(12, 6))
        sns.histplot(col1, kde=True, color='blue', label=label1, alpha=0.5, binwidth=300)
        sns.histplot(col2, kde=True, color='orange', label=label2, alpha=0.5, binwidth=300)
        plt.title('Histogram of Runtimes', fontsize=16)
        plt.xlabel('Runtime', fontsize=14)
        plt.xticks(np.arange(0, max_time_limit+1, 3600), rotation=45)
        plt.ylabel('Frequency',  fontsize=14)
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if path != '':
            plt.savefig(path)
        plt.show()


    def show_runtime_histogram_limited(self, col1, col2, label1, label2, path=''):
        plt.figure(figsize=(12, 6))
        sns.histplot(col1, kde=True, color='blue', label=label1, alpha=0.5, binwidth=300)
        sns.histplot(col2, kde=True, color='orange', label=label2, alpha=0.5, binwidth=300)
    
        plt.title('Histogram of Runtimes (0–7200)', fontsize=16)
        plt.xlabel('Runtime', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
    
        # Limit x-axis to [0, 7200] seconds
        plt.xlim(0, 7200)
        plt.xticks(np.arange(0, 7201, 300), rotation=45)  # ticks every 5 minutes
    
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
        if path != '':
            plt.savefig(path)
    
        plt.show()

    def show_scatter_plot(self, df):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['Actual_Runtime'], y=df['Predicted_Runtime'], label='Predicted vs Actual', color='orange')
        plt.plot([df['Actual_Runtime'].min(), df['Actual_Runtime'].max()],
                 [df['Actual_Runtime'].min(), df['Actual_Runtime'].max()], color='blue', linestyle='--', label='Ideal')
        plt.title('Predicted vs Actual Runtime')
        plt.xlabel('Actual Runtime')
        plt.ylabel('Predicted Runtime')
        plt.legend()
        plt.show()


class Metrics:

    def estimation_accuracy(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
    
        # Avoid division by zero
        mask = (y_true > 0) & (y_pred > 0)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
        ea = np.where(y_pred <= y_true,
                      y_pred / y_true,
                      y_true / y_pred)
    
        return float(np.mean(ea))

    def scaled_mae(self, scaler, y_true, y_pred):          
        y_true_scaled = scaler.transform(np.array(y_true).reshape(-1,1)).reshape(-1)
        y_pred_scaled = scaler.transform(np.array(y_pred).reshape(-1,1)).reshape(-1)
        
        return mean_absolute_error(y_true_scaled, y_pred_scaled)

    def mean_absolute_percentage_error(self, y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return mape
        
    def convert_to_hms(self, time_value, unit="seconds"):
        if unit == "minutes":
            time_value *= 60  # Convert minutes to seconds
        hours, remainder = divmod(time_value, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def print(self, scaler, y_test, y_pred, start, end, len_test):
        mae = mean_absolute_error(y_test, y_pred)
        mae_hhmmss = self.convert_to_hms(int(mae))
        mae_scaled = self.scaled_mae(scaler, y_test, y_pred)
        ea = self.estimation_accuracy(y_test, y_pred)
        mape = self.mean_absolute_percentage_error(y_test, y_pred)

        print("-------------------------------------------")
        print("                 METRICS")
        print("-------------------------------------------")
        print(f"Inference time: {end - start}")
        print(f"Latency:        {(end - start)/len_test}")
        print("-------------------------------------------")
        print(f"MAE:            {mae}")
        print(f"MAE (hh:mm:ss): {mae_hhmmss}")
        print(f"MAE (Scaled):   {mae_scaled}")
        print(f"EA:             {ea}")
        print(f"MAPE:           {mape}")
        print("-------------------------------------------")
