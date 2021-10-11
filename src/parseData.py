import csv
import pickle

for year in ['2019']:
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        filename = './data/yellow_tripdata_' + year + '-' + month + '.csv'
        filenamePkl = './data/yellow_tripdata_' + year + '-' + month + '.plk'
        #data = pickle.load(open(filenamePkl, 'rb'))
        data = []
        with open(filename) as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            count = 0
            for row in datareader:
                if(count == 0):
                    print(row)
                    header = row
                else:
                    #print(row)
                    data.append(row)
                count += 1
        pickle.dump(data, open(filenamePkl, 'wb'))

