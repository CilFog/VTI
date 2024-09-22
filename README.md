# VTI

## Setup
Ensure you are running python 3.12

## Run TrImpute (OUTDATED)
1. Run ```pip instal -r requirements.txt``` in VTI folder
1. Line 24 in TrImpute.py should be changed to the format of the ais csv input ais format.
1. Line 22 in TrImpute.py, if input contains header, add ```next(f) #skips the header```
1. Ensure that timstamp of input is an int or float value
1. In the datasets folder, create a new folder in the input folder called ais_csv
1. Add the csvs' to the ais_csv folder
1. Run TrImpute from the TrImputeFolder with ```python TrImpute.py ais_csv ais_imputation```

## Run GTI (OUTDATED)
1. Run ```pip instal -r requirements.txt``` in VTI folder
1. Line 15 in [Title](../GTI/build_row_graph.py) file, select folder with trajectories txt files
1. Go to the folder with GTI data files and build graph with ``` python build_row_graph.py```
1. Now, ensure that you have the language go install with ```go version```. Install if not
1. Ensure you are in root folder of GTI and run ```go mod init gti```
1. After, run ```go get github.com/RyanCarrier/dijkstra```
1. Then open a new terminal, and run ```go run routing_distance.go 3333```
1. In another terminal, run ```python interpolate_and_refine.py 3333```
1. You find results in the output folder in data



## To Xiushang
1. In general the code you are probably most interested in are located in the grid_layer_module which handles the depth-map file by saving each recorded depth to a postgreSQL db. The graph_construction_module consideres the (cleansed-AIS) data and creates graphs that are divided into cells. The imputation_module then handles the use of these grids based on an input trajectory. The run.py file can handle these three processes, if already have existing AIS data. We have not spent time cleaning up any of the code since we handed in our paper, and it may get a bit messey sometimes. Feel free to reach out to any of us on alfc@netcompany.com or cefo@netcompany.com. 