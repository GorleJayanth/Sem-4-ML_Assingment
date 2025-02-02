import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

column_d = data.iloc[:, 3]
print(f"D = {column_d}")

mean_value = np.mean(column_d)
variance_value = np.var(column_d, ddof=1)  
print(f"the mean of column D is = {mean_value}")
print(f"the variance of column D is = {variance_value}")

data['Date'] = pd.to_datetime(data['Date'])
data['weekday'] = data['Date'].dt.weekday

wednesday_data = data[data['weekday'] == 2]
wednesday_mean_value = wednesday_data['Price'].mean()
print(f"The sample mean for all Wednesdays in the dataset is = {wednesday_mean_value}")

data['Month'] = data['Date'].dt.month
april_data = data[data['Month'] == 4]
april_mean_value = april_data['Price'].mean()
print(f"The sample mean for April in the dataset is = {april_mean_value}")

loss_count = (data['Chg%'] < 0).sum()
total_records = len(data)
loss_probability = loss_count / total_records
print(f"the probability of making loss in the stock is {loss_probability}")

profit_count_wednesday = (wednesday_data['Chg%'] > 0).sum()
profit_probability_wednesday = profit_count_wednesday / total_records
print(f"the probability of making profit in the stock on Wednesday is {profit_probability_wednesday}")

profit_on_wednesday = wednesday_data[wednesday_data['Chg%'] > 0]
conditional_probability_wednesday = len(profit_on_wednesday) / len(wednesday_data)
print(f"The conditional probability of making profit, given that today is Wednesday = {conditional_probability_wednesday}")

data['Day_of_week'] = data['Date'].dt.weekday

sns.scatterplot(x="Day_of_week", y="Chg%", data=data, hue="Day_of_week", palette="hls")
plt.xlabel("Day (of the week)")
plt.show()
