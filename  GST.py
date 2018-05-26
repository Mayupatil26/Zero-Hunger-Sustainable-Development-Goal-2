#importing CSV File and Reading it
import pandas as pd
import csv
import pyspark
from pyspark import SparkContext, SparkConf
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

#spark context created
conf=SparkConf().setAppName("CSE545 Project").set("spark.driver.memory", "12g").set("spark.executor.memory", "6g").set("spark.driver.maxResultSize", "6g")
sc=SparkContext(conf=conf)

#Loading the Climate Dataset
GST_rdd = sc.textFile("GlobalLandTemperaturesByCountry.csv").map(lambda line: line.split(","))

#Filtering out too old records
GST_rdd=GST_rdd.map(lambda x: [w.encode("utf-8") for w in x]).filter(lambda x: x[0] =='dt' or int(x[0][:4])>1960)

#Reformatting the dataset for desired attributes
def reformat(x):
    if x[0]!='dt':
        x[0]=int(x[0][:4])
        
    return [x[3],x[0],x[1]]

GST_rdd2=GST_rdd.map(lambda x: reformat(x))

#COnverting String Values to Integer Values
def conv_x(x):
    if x[0] != "Country":
        if x[2] == "":
            x[2] = str(0)
        x[2] = round(float(x[2]),2)
    return x
            
    
#Call to the conversion function
GST_rdd3=GST_rdd2.map(lambda x: conv_x(x))
print(GST_rdd3.collect())

#Filling in Missing Values
GST_rdd4=GST_rdd3.map(lambda x: (x[0],x[1]))
GST_rdd5=GST_rdd3.map(lambda x: x[2]).collect()

for i in range(len(GST_rdd5)):
    if(i==len(GST_rdd5)-1 and GST_rdd5[i]==0):
        GST_rdd5[i]=round((2*GST_rdd5[i-1]-GST_rdd5[i-2]),2)
    elif(GST_rdd5[i]==0):
        GST_rdd5[i]=round(((GST_rdd5[i-1]+GST_rdd5[i+1])/2),2)
    

GST_rdd5=sc.parallelize(GST_rdd5)

GST_rdd4 = GST_rdd4.zipWithIndex().map(lambda x: (x[1],x[0]))
GST_rdd5 = GST_rdd5.zipWithIndex().map(lambda x: (x[1],x[0]))
GST_rdd6 = GST_rdd4.join(GST_rdd5).map(lambda x: x[1]).groupByKey().mapValues(list)

print(GST_rdd6.collect())

#Take Average of temperature of 12 months to get get temperature for a particular year
def take_average(x):
    avg=0
    if(x[1][0]!="AverageTemperature"):
        avg=round((sum(x[1])/len(x[1])),2)
    else:
        return [x[0][0],x[0][1],x[1][0]]
    
    return [x[0][0],x[0][1],avg]

#call to the take_average function
GST_rdd7=GST_rdd6.map(lambda x: take_average(x))


#Transorming rdd to pandas dataframes for future ML and visualizations
headers = ["Country","Year","Average Temperature"]

GST_rdd7=GST_rdd7.filter(lambda x: x[0]!='Country')
GST_rdd7=GST_rdd7.sortBy(lambda x: (x[0],x[1]))

df = pd.DataFrame(GST_rdd7.collect(), columns=headers)

#reading country code 
ccode=pd.read_csv("codes.csv")

#joining country with country codes
df1=df.join(ccode.set_index('COUNTRY'), on='Country')
df1=df1.dropna(how="any").drop(['GDP (BILLIONS)'], axis=1)

#getting unique countries from dataset
country_unique = df1.Country.unique()

#retriving values for 2013,1998
df2=df1[df1['Year']== 2013]
df3=df1[df1['Year']== 1998]

#dropping columns country and year
df3=df3.drop(['Country','Year'],axis=1)
df3=df3.rename(index=str, columns={"Average Temperature" : "Avg"})
df3=df3.join(df2.set_index('CODE'), on='CODE')
df3=df3.drop(['Year'],axis=1)

#visualizing the graph for temperature change.
data = [ dict(
        type = 'choropleth',
        locations = df3['CODE'],
        z = (df3['Average Temperature']-df3['Avg'])*100/df3['Avg'],
        text = df3['Country'],
        colorscale = [[0.0, 'rgb(84,39,143)'],[0.2, 'rgb(117,107,177)'],[0.4, 'rgb(158,154,200)'],\
            [0.6, 'rgb(188,189,220)'],[0.8, 'rgb(218,218,235)'],[1.0, 'rgb(242,240,247)']],
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
            title = 'Percentage change in Temperature (in degree celsius)'),
      ) ]

layout = dict(
    title = 'Change in Global Surface Temperature',
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
