# Group 23 - Analysing NYC taxi data

This project uses public data set from the NYC TLC website to look for the pick-up and drop off times, trip lengths, average trip speed, surcharges associated with the trip and the number of people taking the taxi through the day to suggest the best time to travel by taxi in NYC. This way, people can plan their trip ahead of time by checking our model which will have predictions of the best time for every day of the year.

### Dependencies ###

Clone repository

Create venv

```
pip install -r requirements.txt
```

The following packages were used:

* pandas
* numpy
* shapely
* SQLAlchemy
* descartes
* matplotlib
* pyshp
* sklearn


### Folder Organization ###

In the main folder:

 - [Group_23_NYC_taxi.pdf](Group_23_NYC_taxi.pdf) -- presentation slides in pdf format
 
 - [Group23_Assignment7.ipynb](Group23_Assignment7.ipynb) -- Assignment test cases as a  Notebook file
 
 - VizualisationNotebooks : Visualization used in the presentation are in the Presentation_Visualization folder and the visualizations done post-presentation are in the heat map  visualization folder.
 - src : contains python source codes that we used to figure the dataset out and experiment on

### Dataset ###

All the data we used can be found in the following link: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Each month has about 1 million data points. 12 months of 2019 contains approximately 12 million data points.

### Usage ###

We have written code to directly download the csv files as you run the code using the python standard module [urllib.request](https://docs.python.org/3/library/urllib.request.html) because each month's csv file is approximately 700MB.
Feel free to download the trip_records.csv file from the NYC TLC webpage incase there is an error in accessing the url using the urllib module and store in the proper directory to access it or the code accesses the file directly without downloading it from the webpage specified.


### Visualizations ###

All the vizualisations use in the presentation can be found in the [Final_Viz_team23.ipynb](VisualizationNotebooks/Presentation_Visualization/Final_Viz_team23.ipynb) file, and the images we generated are stored alongside in the Presentation_Visualization folder under Visualization notebooks folder. The visualization done after time of presentation are added in the heat_map folder in the same folder.
