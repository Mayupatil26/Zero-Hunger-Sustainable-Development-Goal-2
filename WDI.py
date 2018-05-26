#importing CSV File and Reading it
import pandas as pd
import csv
import pyspark
from pyspark import SparkContext, SparkConf
import numpy as np
import tensorflow as tf
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, zscore

#Starting the Spark Session
conf=SparkConf().setAppName("CSE545 Project").set("spark.driver.memory", "12g").set("spark.executor.memory", "6g").set("spark.driver.maxResultSize", "6g")
sc=SparkContext(conf=conf)

#Loading WDI Dataset
WDI_rdd1 = sc.textFile("WDI_GDP_Growth.csv").map(lambda line: line.split(",")).filter(lambda line: len(line)>1)

#collecting countries codes
ccode = WDI_rdd1.map(lambda x: x[1]).filter(lambda x: x!="Country Code")
ccode_list = ccode.collect()

#Filtering required data
WDI_rdd2 = WDI_rdd1.map(lambda x: [x[0],x[29:60]])

#Converting String Values to Integer Values
def conv_x(x):
    growth=[]
    count=0
    temp=[]
    if x[0] != "Country Name":
        for num in x[1]:
            if num == '':
                temp.append(0)
                count+=1
                if(count==8):
                    growth.append(temp)
                    temp=[]
                    count=0
            else:
                temp.append(round(float(num),2))
                count+=1
                if(count==8):
                    growth.append(temp)
                    temp=[]
                    count=0
        growth.append(temp)
    else:
        growth=x[1]
    return [x[0],growth]

#Calculating growth over 8 years
def calc_growth8(x):
    wdi=[]
    if x[0] != "Country Name":
        for l in x[1]:
            gr=0
            for i in range(len(l)):
                gr=round(float(l[i] + gr + (l[i]*gr/100)),2)
            wdi.append(gr)
    else:
        wdi=x[1]
    return [x[0],wdi]


WDI_rdd3=WDI_rdd2.map(lambda x: conv_x(x)).map(lambda x: calc_growth8(x))

#Transforming rdd to tensors for applying Machine Learning Models
X_wdi=WDI_rdd3.filter(lambda x: x[0]!="Country Name").map(lambda x: x[1])
X_wdi=np.array(X_wdi.collect())

#collecting countries list
countries_list = WDI_rdd3.filter(lambda x: x[0]!="Country Name").map(lambda x: x[0])
countries = countries_list.collect()

#calculating Beta
def calc_beta(betas, X_test, y_test):
    y_pred = np.matmul(X_test, betas)[:,0]
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    
#rigde Regression
def RidgeRegression(X_wdi,penalty_value = 1.0, learning_rate = 0.000000001, n_epochs = 100):

    #Dividing into training and test data
    offset = int(int(X_wdi.shape[0]) * 0.8)
    X_wdi_tf_test, Y_wdi_tf_test = X_wdi[offset:,:3], X_wdi[offset:,3:]
    X_wdi_tf, Y_wdi_tf = X_wdi[:offset,:3], X_wdi[:offset,3:]

    # Conversion to tensors
    X_wdi_tf = tf.constant(X_wdi_tf, dtype=tf.float32, name="WDI_X")
    Y_wdi_tf = tf.constant(Y_wdi_tf.reshape(-1,1), dtype=tf.float32, name="WDI_Y")
    Xt_wdi_tf = tf.transpose(X_wdi_tf)
    penalty = tf.constant(1.0, dtype=tf.float32, name="penalty")
    I = tf.constant(np.identity(int(X_wdi_tf.shape[1])), dtype=tf.float32, name="I")
    beta = tf.Variable(tf.random_uniform([int(X_wdi_tf.shape[1]), 1], -1., 1.), name = "beta")
    y_pred = tf.matmul(X_wdi_tf, beta, name="predictions")
    penalizedCost = tf.reduce_sum(tf.square(Y_wdi_tf - y_pred)) + penalty * tf.reduce_sum(tf.square(beta))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(penalizedCost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch %10 == 0: #print debugging output
                print("Epoch", epoch, "; penalizedCost =", penalizedCost.eval())
            sess.run(training_op)
        #done training, get final beta: 
        best_beta = beta.eval()
    print(best_beta)
    calc_beta(best_beta, X_wdi_tf_test, Y_wdi_tf_test)
    return best_beta
#best Beat value
best_beta_val = RidgeRegression(X_wdi,.1)

#calculating 2024 year value
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
WDI_2024 = tf.matmul(X_wdi[:,1:], best_beta_val, name="predictions")
s = WDI_2024.eval()

#close the session to release resources
sess.close()

X_wdi_df = pd.DataFrame(X_wdi)
X_wdi_df = X_wdi_df.rename(index=str, columns={0: "1992", 1: "2000",2: "2008",3: "2015"})
X_wdi_df['2024'] = s


#calculating 2032 year value
X_wdi_2024 = np.array(X_wdi_df)
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
WDI_2032 = tf.matmul(X_wdi_2024[:,2:], best_beta_val, name="predictions")
s = WDI_2032.eval()

#close the session to release resources
sess.close()

X_wdi_2024 = pd.DataFrame(X_wdi_2024)
X_wdi_2024 = X_wdi_2024.rename(index=str, columns={0: "1992", 1: "2000",2: "2008",3: "2015",4: "2024"})
X_wdi_2024['2032'] = s


#calculating 2040 year value
X_wdi_2032 = np.array(X_wdi_2024)
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
WDI_2040 = tf.matmul(X_wdi_2032[:,3:], best_beta_val, name="predictions")
s = WDI_2040.eval()

#close the session to release resources
sess.close()

X_wdi_2032 = pd.DataFrame(X_wdi_2032)
X_wdi_2032 = X_wdi_2032.rename(index=str, columns={0: "1992", 1: "2000",2: "2008",3: "2015",4: "2024",5:"2032"})
X_wdi_2032['2040'] = s

#rouding off to 2 decimal places
X_wdi_2032 = X_wdi_2032.round(2)

#country and country code dataframe
countr = pd.DataFrame({"countries": countries})
countr_code = pd.DataFrame({"ccode_list": ccode_list})

#visualization for year - 1992
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['1992'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 1992',
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


#visualization for year - 2000
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2000'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2000',
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


#visualization for year - 2008
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2008'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2008',
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


#visualization for year - 2015
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2015'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2015',
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


#visualization for year - 2024
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2024'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2024',
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

#visualization for year - 2032
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2032'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2032',
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

#visualization for year - 2040
data = [ dict(
        type = 'choropleth',
        locations = countr_code['ccode_list'],
        z = X_wdi_2032['2040'],
        text = countr["countries"],
        colorscale = [[0.0, 'rgb(254,235,226)'],[0.2, 'rgb(252,197,192)'],[0.4, 'rgb(250,159,181)'],\
            [0.6, 'rgb(247,104,161)'],[0.8, 'rgb(197,27,138)'],[1.0, 'rgb(122,1,119)']],
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
            title = '% Growth in GDP'),
      ) ]

layout = dict(
    title = 'World Development Indicator - 2040',
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
