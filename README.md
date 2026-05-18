# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18.05.2026



### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```

#Developed By: HAREESH R
#Reg No: 212223230068

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load the dataset,perform data exploration
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'],index_col='Month')

data_monthly = data.resample('MS').sum()   #Month start

data_monthly.head()
data_monthly.plot()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)
scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"RMSE for test predictions: {rmse}")
np.sqrt(scaled_data.var()),scaled_data.mean()

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```
### OUTPUT:
## SCALED_DATA PLOT:


<img width="614" height="416" alt="image" src="https://github.com/user-attachments/assets/fb67f361-5a84-435e-bc77-d359b7f32ba0" />

## DEMCOMPOSED PLOT:

<img width="659" height="447" alt="image" src="https://github.com/user-attachments/assets/8057350d-987c-48bb-bda2-7357339c96c6" />


## TEST_PREDICTION:

<img width="678" height="419" alt="image" src="https://github.com/user-attachments/assets/c4c14fc5-4989-4bba-8e84-a0a00f5e7c3e" />

<img width="628" height="430" alt="image" src="https://github.com/user-attachments/assets/fc782684-a53d-4e9d-9cfb-4f1f8119b3a4" />

## RMSE VALUE:

<img width="352" height="47" alt="image" src="https://github.com/user-attachments/assets/cd467093-6f2d-4b5e-9065-0e288d237259" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
