#import all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------

columnheadings = ['ID', 'Diagnosis',
'radiusM', 'textureM', 'perimeterM', 'areaM', 'smoothnessM', 'compactnessM', 'concavityM', 'concave pointsM', 'symmetryM', 'fractal_dimensionM',
'radiusSE', 'textureSE', 'perimeterSE', 'areaSE', 'smoothnessSE', 'compactnessSE', 'concavitySE', 'concave pointsSE', 'symmetrySE', 'fractal_dimensionSE',
'radiusW', 'textureW', 'perimeterW', 'areaW', 'smoothnessW', 'compactnessW', 'concavityW', 'concave pointsW', 'symmetryW', 'fractal_dimensionW']

dtset = pd.DataFrame(pd.read_csv('/home/dhanvee/Documents/cAIncer/WDBC/WDBC.csv', names = columnheadings, header = None))    #import the .csv file

plotset = dtset.drop('ID', axis=1)
plotset.columns = (range(1,32))
plotset = plotset.replace(to_replace='B', value='Green')
plotset = plotset.replace(to_replace='M', value='Red')

#---------------------------------------------------------------------------------

def plotScatter(df, xaxis, yaxis, diagnosis, columnheadings):
	df.plot.scatter(x=xaxis, y=yaxis, c=diagnosis, s=7, alpha=0.3)
	filename = str(columnheadings[xaxis])+"_"+str(columnheadings[yaxis])+".png"
	plt.savefig("/home/dhanvee/Documents/cAIncer/PlotTrial2/" + filename, dpi = 250)

def individualGraph(df, xaxis, yaxis, diagnosis, columnheadings):
	for xaxis in range(2,12):
		for yaxis in range ((xaxis+1), 12):
			plotScatter(plotset, xaxis, yaxis, list(plotset[1]), columnheadings)
			plt.close()


print("Done!")