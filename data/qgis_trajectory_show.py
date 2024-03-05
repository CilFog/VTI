import os

# open qgis -> plugins -> python
# update basefolder to a folder with txt files
# ensure header in the txt files
# paste script to qgis
basefolder = 'path-to-folder'
epsg_code=4326
xfield, yfield = 'longitude', 'latitude'
delimiter = ','

layers=[]
for root, folder, files in os.walk(basefolder):
    for file in files:
        fullPath = os.path.join(root, file)
        if os.path.isfile(fullPath) and fullPath.endswith('.txt'):
            uri = "file://{}?delimiter='{}'&xField={}&yField={}&crs=epsg:{}".format(fullPath, delimiter, xfield, yfield, epsg_code)
            vlayer = QgsVectorLayer(uri, os.path.basename(file).split('.')[0] , "delimitedtext")
            layers.append(vlayer)

QgsProject.instance().addMapLayers(layers)