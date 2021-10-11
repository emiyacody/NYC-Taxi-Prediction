import pickle
import numpy as np
import pickle
import sys
import datetime
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 
# 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 
# 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']

if len(sys.argv) != 4:
    print('incorrect number of arguments')
    print('Usage: ')
    print('1st argument: Month')
    print('2nd argument: Day')
    print('3rd argument: Hour')
    exit()

month = sys.argv[1]
day = sys.argv[2]
hour = float(sys.argv[3])
year = '2019'

filenamePkl = './data/yellow_tripdata_' + year + '-' + month + '.plk'
data = pickle.load(open(filenamePkl, 'rb'))

dates = []
extra = []
mta_tax = []
tip_amount = []
tolls_amount = []
improvement_surcharge = []
congestion_surcharge = []
total_amount = []

for i in range(len(data)):
    try:
        extra.append(float(data[i][11]))
    except:
        extra.append(0.0)
    try:
        improvement_surcharge.append(float(data[i][15]))
    except:
        improvement_surcharge.append(0.0)
    try:
        congestion_surcharge.append(float(data[i][17]))
    except:
        congestion_surcharge.append(0.0)
    try:
        total_amount.append(float(data[i][16]))
    except:
        total_amount.append(-1.0)
    try:
        date_time = datetime.datetime.strptime(data[i][1], '%Y-%m-%d %H:%M:%S')
        dates.append(date_time)
    except:
        dates.append('')

extra = np.asarray(extra, dtype=np.float64)
improvement_surcharge = np.asarray(improvement_surcharge, dtype=np.float64)
congestion_surcharge = np.asarray(congestion_surcharge, dtype=np.float64)

total_surcharges = extra + improvement_surcharge + congestion_surcharge

total_surcharges_percent = []
for i in range(len(total_amount)):
    if(total_amount[i] > 0):
        total_surcharges_percent.append(total_surcharges[i]/total_amount[i])
    else:
        total_surcharges_percent.append(10)
total_surcharges_percent = np.asarray(total_surcharges_percent, dtype=np.float64)

final_hours = []
final_total_surcharges_percent = []

for i in range(len(dates)):
    if(isinstance(dates[i], datetime.date) and total_surcharges_percent[i] <= 0.5):
        data_minute = dates[i].minute
        data_hour = dates[i].hour
        data_day = dates[i].day
        if(int(data_day) == int(day)):
            final_hours.append(float(data_hour) + float(data_minute)/60.0)
            final_total_surcharges_percent.append(total_surcharges_percent[i])

final_hours = np.asarray(final_hours, dtype=np.float64)
final_total_surcharges_percent = np.asarray(final_total_surcharges_percent, dtype=np.float64)

plt.figure(figsize=(20, 10))
plt.scatter(final_hours, final_total_surcharges_percent, color='black', label='True')
plt.title('Available True Data', fontsize=28, fontweight='bold')
plt.xlabel('Hour', fontsize=28, fontweight='bold')
plt.ylabel('% Of Total Payment That is a Surcharge', fontsize=28, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.savefig('./plots/trueData.png')

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_rbf.fit(final_hours.reshape(-1, 1), final_total_surcharges_percent)
print(svr_rbf.predict(np.array([hour]).reshape(1, -1))[0])

sample_hours = []
for i in range(1, 24):
    for j in range(100):
        sample_hours.append(i + random.random())

sample_hours = np.asarray(sample_hours, dtype=np.float64)

predicts = svr_rbf.predict(sample_hours.reshape(-1, 1))

final_hours = np.asarray(sample_hours, dtype=np.float64)
final_total_surcharges_percent = np.asarray(predicts, dtype=np.float64)

plt.scatter(final_hours, final_total_surcharges_percent, color='blue', label='Predicted')
plt.title('Added Predicted Data', fontsize=28, fontweight='bold')
plt.xlabel('Hour', fontsize=28, fontweight='bold')
plt.ylabel('% Of Total Payment That is a Surcharge', fontsize=28, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend()
plt.savefig('./plots/predictData.png')
