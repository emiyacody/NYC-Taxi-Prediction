import pandas as pd
import matplotlib.pyplot as plt
import shapefile
plt.style.use('ggplot')
import numpy as np
from itertools import chain
from collections import Counter
PU_allstar =[]
DO_allstar =[]
for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    filename = 'yellow_tripdata_2019-'+ month +'.csv'
    with open(filename) as csvfile:
        df =pd.read_csv(filename)
        #print(df.head())
        PU_location = df['PULocationID']
        DO_location = df['DOLocationID']
        PU_count = df['PULocationID'].value_counts().sort_values()
        DO_count = df['DOLocationID'].value_counts().sort_values()
        dq = pd.DataFrame()
        dq['PUcount'] =PU_count
        dq['DOcount'] = DO_count
        dq.reset_index(inplace=True)
        dq.rename(columns={'index':'LocationID'}, inplace=True)
        PUtop5 = dq.sort_values(by = ['PUcount'], ascending=False).head(5)
        PU_TOP = PUtop5.LocationID.tolist()
        PU_allstar.append(PU_TOP)
        DOtop5 = dq.sort_values(by=['DOcount'], ascending=False).head(5)
        DO_TOP = DOtop5.LocationID.tolist()
        DO_allstar.append(DO_TOP)
#plot map#
a = list(chain.from_iterable(PU_allstar))
print(Counter(a))
b = list(chain.from_iterable(DO_allstar))
print(Counter(b))
draw_PU =list(Counter(a).keys())
draw_DO =list(Counter(b).keys())
tone_PU =[]
tone_DO = []
#set color range
for val in list(Counter(a).values()):
    if val >11:
        color = '#dc143c'
        tone_PU.append(color)
    elif 9<=val<=11:
        color = '#fee391'
        tone_PU.append(color)
    elif 5<= val <=8:
        color = '#fe9929'
        tone_PU.append(color)
    elif 0<val<=4:
        color = '#e97451'
        tone_PU.append(color)
print(tone_PU)
for val in list(Counter(b).values()):
    if val >11:
        color = '#008000'
        tone_DO.append(color)
    elif 9<=val<=11:
        color = '#3cb371'
        tone_DO.append(color)
    elif 5<= val <=8:
        color = '#ace1af'
        tone_DO.append(color)
    elif 0<val<=4:
        color = '#98FB98'
        tone_DO.append(color)
print(tone_DO)
#read map
sf = shapefile.Reader("taxi_zones.shp")
figsize=(10,10)
x_lim=None
y_lim =None

#PUmap
f1 = plt.figure(1)
fig, ax = plt.subplots(figsize=figsize)
ax.set_title("Most Pick-Up zones")
ocean = (80 / 256, 160 / 256, 210 / 256)
ax.set_facecolor(ocean)
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y,'k')

for id in draw_PU:
    shape_ex = sf.shape(id-1)
    x_lon = np.zeros((len(shape_ex.points), 1))
    y_lat = np.zeros((len(shape_ex.points), 1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    ax.fill(x_lon, y_lat, tone_PU[draw_PU.index(id)])


#DOmap
f2 = plt.figure(2)
fig, ax = plt.subplots(figsize=figsize)
ax.set_title("Most Drop-Off zones")
ocean = (80 / 256, 160 / 256, 210 / 256)
ax.set_facecolor(ocean)
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x, y,'k')
for id in draw_DO:
    shape_ex = sf.shape(id-1)
    x_lon = np.zeros((len(shape_ex.points), 1))
    y_lat = np.zeros((len(shape_ex.points), 1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    ax.fill(x_lon, y_lat, tone_DO[draw_DO.index(id)])
plt.show()