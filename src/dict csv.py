import csv
from collections import defaultdict
with open('taxi+_zone_lookup.csv', 'r') as rf:
    reader = csv.reader(rf)
    mydict = defaultdict(list)
    mydict_zone = defaultdict(list)
    borough_vs_zone = defaultdict(list)
    for rows in reader:
        for cols in reader: 
            mydict[cols[1]].append(cols[0])
            mydict_zone[cols[0]].append(cols[2])
            borough_vs_zone[cols[1]].append(cols[2])
    


##this dict shows locationIDs corresponded to their Borough
print(mydict)

borough = mydict.keys()
location_ID = mydict.values()
#print(borough)
#print(location_ID)

##this dict tells what zone the locationID belongs to.
print(mydict_zone)
loc_zone= mydict_zone.keys()
zone = mydict_zone.values()

##this dict tells what borough the zones belong to. 
print(borough_vs_zone)
#print(borough_vs_zone['Queens'])
