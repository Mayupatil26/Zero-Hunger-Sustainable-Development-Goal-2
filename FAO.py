#importing CSV File and Reading it
import pandas as pd
import numpy as np
import csv
import scipy
from scipy.stats import pearsonr
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

FAO_data = pd.read_csv("FoodBalanceSheets_E_All_Data.csv", sep=',', encoding='latin-1')

#Dropping Useless Columns
FAO_data_col_drop=FAO_data.drop(['Area Code','Item Code','Element Code','Year Code','Flag'],axis=1)
#dropping duplicates
FAO_data_dup_drop=FAO_data_col_drop.drop_duplicates()

#grouping by area,element,year
FAO_data_group = FAO_data_dup_drop.groupby(['Area','Element','Year']).size().reset_index().rename(columns={0:'count'})
FAO_data_group_drCount = FAO_data_group.drop('count',axis=1)

#Converting items into features using grouping
l = []
cnt = 0
for index,row in FAO_data_group_drCount.iterrows():
    cnt += 1
    d = dict()
    ds = FAO_data_dup_drop[(FAO_data_dup_drop['Area'] == row[0])]
    ds1 = ds[ds['Element'] == row[1]]
    ds2 = ds1[ds1['Year'] == row[2]]
    d['Area'] = row[0]
    d['Element'] = row[1]
    d['Year'] = row[2]
    d['Unit'] = ds2['Unit'].head(1).values[0]
    for ind, r in ds2.iterrows():
        d[r['Item']] = r['Value']
    l.append(d)
    if cnt%1000 == 0:
        print(cnt, " :done")

#converting list to dataframe
FA0_dataset = pd.DataFrame(l)

#saving to csv file
FA0_dataset.to_csv("FA0_dataset.csv", index = False)

#reading from csv file
FA0_dataset = pd.read_csv("FA0_dataset.csv")

#reading data from .csv file 
colnames = ['Alcohol, Non-Food', 'Alcoholic Beverages', 'Animal Products', 'Animal fats', 'Apples and products', 'Aquatic Animals, Others', 'Aquatic Plants', 'Aquatic Products, Other', 'Bananas', 'Barley and products', 'Beans', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented', 'Bovine Meat', 'Butter, Ghee', 'Cassava and products', 'Cephalopods', 'Cereals - Excluding Beer', 'Cereals, Other', 'Citrus, Other', 'Cloves', 'Cocoa Beans and products', 'Coconut Oil', 'Coconuts - Incl Copra', 'Coffee and products', 'Cottonseed', 'Cottonseed Oil', 'Cream', 'Crustaceans', 'Dates', 'Demersal Fish', 'Eggs', 'Fats, Animals, Raw', 'Fish, Body Oil', 'Fish, Liver Oil', 'Fish, Seafood', 'Freshwater Fish', 'Fruits - Excluding Wine', 'Fruits, Other', 'Grand Total', 'Grapefruit and products', 'Grapes and products (excl wine)', 'Groundnut Oil', 'Groundnuts (Shelled Eq)', 'Honey', 'Infant food', 'Lemons, Limes and products', 'Maize Germ Oil', 'Maize and products', 'Marine Fish, Other', 'Meat', 'Meat, Aquatic Mammals', 'Meat, Other', 'Milk - Excluding Butter', 'Millet and products', 'Miscellaneous', 'Molluscs, Other', 'Mutton & Goat Meat', 'Nuts and products', 'Oats', 'Offals', 'Offals, Edible', 'Oilcrops', 'Oilcrops Oil, Other', 'Oilcrops, Other', 'Olive Oil', 'Olives (including preserved)', 'Onions', 'Oranges, Mandarines', 'Palm Oil', 'Palm kernels', 'Palmkernel Oil', 'Peas', 'Pelagic Fish', 'Pepper', 'Pigmeat', 'Pimento', 'Pineapples and products', 'Plantains', 'Potatoes and products', 'Poultry Meat', 'Pulses', 'Pulses, Other and products', 'Rape and Mustard Oil', 'Rape and Mustardseed', 'Rice (Milled Equivalent)', 'Ricebran Oil', 'Roots, Other', 'Rye and products', 'Sesame seed', 'Sesameseed Oil', 'Sorghum and products', 'Soyabean Oil', 'Soyabeans', 'Spices', 'Spices, Other', 'Starchy Roots', 'Stimulants', 'Sugar & Sweeteners', 'Sugar (Raw Equivalent)', 'Sugar Crops', 'Sugar beet', 'Sugar cane', 'Sugar non-centrifugal', 'Sunflower seed', 'Sunflowerseed Oil', 'Sweet potatoes', 'Sweeteners, Other', 'Tea (including mate)', 'Tomatoes and products', 'Treenuts', 'Vegetable Oils', 'Vegetables', 'Vegetables, Other', 'Vegetal Products', 'Wheat and products', 'Wine', 'Yams']

#pearson correlation plotting
corrs = []
cvalue = 0
ccolname = []
ncols = 119

#get the first column from the dataframe
for i in range(0, ncols):
    data1 = np.array(FA0_dataset[colnames[i]])
    cvalue = cvalue + 1
    ccolname.append(colnames[i])
    
    #getting indexes where value is NaN for column1
    data1_ind = np.where(np.isnan(data1))
    
    #get second column from dataset
    for j in range(0, ncols):
        data2 = np.array(FA0_dataset[colnames[j]])
        
        #deleting indexes if either has NaN values 
        new_data1 = np.delete(data1, data1_ind)
        new_data2 = np.delete(data2, data1_ind)
        
        #getting indexes where value is NaN for column2
        new_data2_ind = np.where(np.isnan(new_data2))
        data2_withoutNan = np.delete(new_data2, new_data2_ind)
        data1_withoutNan = np.delete(new_data1, new_data2_ind)
        
        #calculating pearson correlation
        corr, n_corr = scipy.stats.pearsonr(data1_withoutNan,data1_withoutNan)
        
        #appending the correlation into array
        corrs.append(corr)

a = np.array(corrs).reshape(cvalue,cvalue)
ax = sns.heatmap(a)
ax.set_yticklabels(reversed(colnames))
ax.set_xticklabels(colnames)
plt.show()

#filling Nan values with 0
FA0_dataset = FA0_dataset.fillna(0)

#reading country codes
ccodes = pd.read_csv("ccodes.csv")

#mapping country with country code
countries = pd.DataFrame({'COUNTRY':FA0_dataset['Area'].unique()})
ccodes = pd.read_csv("ccodes.csv")
result = pd.merge(countries, ccodes, on='COUNTRY')
result = result.drop(['GDP (BILLIONS)'],axis=1)


#calculating the values of Domestic supply for each area
l=[]
for index, row in result.iterrows():
    FA0_dataset1 = FA0_dataset[FA0_dataset['Area']==row[0]]
    FA0_dataset2 = FA0_dataset1[FA0_dataset1['Element']=='Domestic supply quantity']
    asd = FA0_dataset2.sum(axis=1)
    qwe = asd.tolist()
    l.append(qwe)


#collecting the Domestic supply values in a list
j = []
for i in range(0,len(l)):
    a = l[i][0]
    j.append(a)


#creating the dataframe from list
qwerty = pd.DataFrame(j)


#visualizing the graph
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = qwerty[0],
        text = result["COUNTRY"],
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'GHI'),
      ) ]

layout = dict(
    title = 'Global Hunger Index - 2024',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )











