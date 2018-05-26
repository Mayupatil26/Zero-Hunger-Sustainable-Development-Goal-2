#importing CSV File and Reading it
import pandas as pd
import csv
import pyspark
from pyspark import SparkContext, SparkConf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

#Loading GHI Dataset
GHI_rdd1 = sc.textFile("002_AppendixD.tsv").map(lambda line: line.split("\t"))

#Deleting records with more than 1 missing values
def count_x(x):
    cnt = 0
    for i in range(0,len(x)):
        if x[i] == " -":
            cnt += 1
    if cnt > 1:
        return False
    return True
    
GHI_rdd1 = GHI_rdd1.filter(lambda line: len(line)>1).map(lambda x: [w.replace("\"","") for w in x]).filter(lambda x: count_x(x))

#Converting Strings to floats wherever necessary
def conv_x(x):
    if x[0] != "Country":
        x[1] = float(x[1])
        x[2] = float(x[2])
        x[3] = float(x[3])
        x[4] = float(x[4])
    
    return x

#Transforming <5 values into integer values
def rep_5(x):
    a = []
    cnt = 0
    for i in range(0,len(x)):
        if x[i] == "<5":
            x[i] = 4 - cnt
            x[i] = str(x[i])
            cnt += 1
        a.append(x[i])
        
    return a

#Filling in Missing Values based on the future differences
def fill_x(x):
    if x[1] == " -":
        s = float(x[2]) - float(x[3])
        d = float(x[3]) - float(x[4])
        j = (s + d)/2
        x[1] = str(round(j + float(x[2]),1))
        
    return x

GHI_rdd2 = GHI_rdd1.map(lambda x: rep_5(x)).map(lambda x: fill_x(x)).map(lambda x: conv_x(x))

#Countries whose records are there in the current dataset
GHI_rdd2_unique_val = GHI_rdd2.map(lambda x: x[0]).filter(lambda x: x!="Country").distinct().collect()

#Loading other values dataset
GHI_rdd3 = sc.textFile("001_AppendixC.tsv").map(lambda line: line.split("\t")).filter(lambda line: len(line)>1).map(lambda x: [w.replace("\"","") for w in x])

#Filtering values which have corresponding record in GHI dataset
GHI_rdd3 = GHI_rdd3.filter(lambda x: x[0] in GHI_rdd2_unique_val)

#Formatting dataset
GHI_rdd3 = GHI_rdd3.map(lambda x: [x[0],[x[1],x[5],x[9],x[13]],[x[2],x[6],x[10],x[14]],[x[3],x[7],x[11],x[15]],[x[4],x[8],x[12],x[16]]])

#CHeking for the number ofmissing values
def cnt(x):
    a = []
    for i in range(0,len(x)):
        c = 0
        for j in range(0,len(x[i])):
            if x[i][j] == "-":
                c += 1
        a.append(c)
    return a

#Removing the records with at least 2 missing values
def mor_2(x):
    for i in range(0,len(x[1])):
        if x[1][i] == 2:
            return x[0]
        
GHI_rdd3_toRemove = GHI_rdd3.map(lambda x: (x[0],cnt(x[1:5]))).map(lambda x: mor_2(x)).filter(lambda x: x!=None).collect()

#Updating GHI and values datasets for the final filtered results
GHI_rdd4 = GHI_rdd3.filter(lambda x: x[0] not in GHI_rdd3_toRemove)
GHI_rdd2 = GHI_rdd2.filter(lambda x: x[0] not in GHI_rdd3_toRemove)

#Filling in missing values according to the GHI value
def cal_x0(x,ghi_rdd):
    if x[0] != "Country":
        for j in range(0,len(ghi_rdd.value)):
            if x[0] == ghi_rdd.value[j][0]:
                ghis = ghi_rdd.value[j]
        for i in range(1,len(x)):
            ghi = ghis[i]
            if x[i][0] == "-":
                x[i][1] = round(float(x[i][1]),1)
                x[i][2] = round(float(x[i][2]),1)
                x[i][3] = round(float(x[i][3]),1)
                x[i][0] = round(3*(ghi - (x[i][3]/3) - (x[i][2]/6) - (x[i][1]/6)),1)
            else:
                x[i][0] = round(float(x[i][0]),1)
                x[i][1] = round(float(x[i][1]),1)
                x[i][2] = round(float(x[i][2]),1)
                x[i][3] = round(float(x[i][3]),1)
    return x


def conv_listx(x):
    a = [x[0]]
    for i in range(1,5):
        for j in range(0,4):
            a.append(x[i][j])
    return a

#Call to fill in the missing values
GHI_rdd2_broadcast=sc.broadcast(GHI_rdd2.collect())
GHI_rdd4 = GHI_rdd4.map(lambda x: cal_x0(x,GHI_rdd2_broadcast)).map(lambda x: conv_listx(x))

#calculating Beta value
def calc_beta(betas, X_test, y_test):
    y_pred = np.matmul(X_test, betas)[:,0]
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    
# Ridge regression
def RidgeRegression(X, penalty_value = 1.0, learning_rate = 0.00000000001, n_epochs = 100):

    #Dividing into training and test data
    offset = int(int(X.shape[0]) * 0.9)
    X_test, Y_test = X[offset:,:3], X[offset:,3:]
    X_tf, Y_tf = X[:offset,:3], X[:offset,3:]

    # Conversion to tensors
    X_tf = tf.constant(X_tf, dtype=tf.float32, name="GHI_X")
    Y_tf = tf.constant(Y_tf.reshape(-1,1), dtype=tf.float32, name="GHI_Y")
    Xt_tf = tf.transpose(X_tf)
    penalty = tf.constant(1.0, dtype=tf.float32, name="penalty")
    I = tf.constant(np.identity(int(X_tf.shape[1])), dtype=tf.float32, name="I")
    beta = tf.Variable(tf.random_uniform([int(X_tf.shape[1]), 1], -1., 1.), name = "beta")
    y_pred = tf.matmul(X_tf, beta, name="predictions")
    penalizedCost = tf.reduce_sum(tf.square(Y_tf - y_pred)) + penalty * tf.reduce_sum(tf.square(beta))
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
    calc_beta(best_beta, X_test, Y_test)
    return best_beta


#Transforming rdd to tensors for applying Machine Learning Models
X_ghi=GHI_rdd2.map(lambda x: x[1:]).filter(lambda x: x[0]!="GHI1992")
X_ghi=np.array(X_ghi.collect())
print(X_ghi)

#best value of Beta
best_beta_val = RidgeRegression(X_ghi, 1)

#session started
#calculating for year 2024 - GHI
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
X_ghi_2024 = tf.matmul(X_ghi[:,1:], best_beta_val, name="predictions")
s = X_ghi_2024.eval()
#close the session to release resources
sess.close()

X_ghi_df = pd.DataFrame(X_ghi)
X_ghi_df = X_ghi_df.rename(index=str, columns={0: "GHI1992", 1: "GHI2000",2: "GHI2008",3: "GHI2015"})
X_ghi_df['GHI2024'] = s

#calculating for year 2032 - GHI
X_ghi_2024 = np.array(X_ghi_df)
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
X_ghi_2032 = tf.matmul(X_ghi_2024[:,2:], best_beta_val, name="predictions")
s = X_ghi_2032.eval()
#close the session to release resources
sess.close()

X_ghi_2024 = pd.DataFrame(X_ghi_2024)
X_ghi_2024 = X_ghi_2024.rename(index=str, columns={0: "GHI1992", 1: "GHI2000",2: "GHI2008",3: "GHI2015",4: "GHI2024"})
X_ghi_2024['GHI2032'] = s

#calculating for year 2040 - GHI
X_ghi_2032 = np.array(X_ghi_2024)
sess = tf.InteractiveSession()
best_beta_val = np.array(best_beta_val, dtype=np.float64)
X_ghi_2040 = tf.matmul(X_ghi_2032[:,3:], best_beta_val, name="predictions")
s = X_ghi_2040.eval()

#close the session to release resources
sess.close()

X_ghi_2032 = pd.DataFrame(X_ghi_2032)
X_ghi_2032 = X_ghi_2032.rename(index=str, columns={0: "GHI1992", 1: "GHI2000",2: "GHI2008",3: "GHI2015",4: "GHI2024",5: "GHI2032"})
X_ghi_2032['GHI2040'] = s

#rounding off 2 decimal places
X_ghi_2032 = X_ghi_2032.round(2)

#For Un parameter
X_param1=GHI_rdd4.map(lambda x: [x[1],x[5],x[9],x[13]]).filter(lambda x: x[0]!="UN9193")
X_param1=np.array(X_param1.collect())

best_beta_val_UN1 = RidgeRegression(X_param1, 1)

#calculating for year 2024 - UN
sess = tf.InteractiveSession()
best_beta_val_UN1 = np.array(best_beta_val_UN1, dtype=np.float64)
X_param1_2024 = tf.matmul(X_param1[:,1:], best_beta_val_UN1, name="predictions")
s = X_param1_2024.eval()

#close the session to release resources
sess.close()

X_param1_df = pd.DataFrame(X_param1)
X_param1_df = X_param1_df.rename(index=str, columns={0: "UN9193", 1: "UN9901",2: "UN0709",3: "UN1416"})
s[s < 0] = 0
X_param1_df['UN2325'] = s

#calculating for year 2032 - UN
X_param1_df_2024 = np.array(X_param1_df)
sess = tf.InteractiveSession()
best_beta_val_UN1 = np.array(best_beta_val_UN1, dtype=np.float64)
X_param1_2032 = tf.matmul(X_param1_df_2024[:,2:], best_beta_val_UN1, name="predictions")
s = X_param1_2032.eval()

#close the session to release resources
sess.close()

X_param1_df_2024 = pd.DataFrame(X_param1_df_2024)
X_param1_df_2024 = X_param1_df_2024.rename(index=str, columns={0: "UN9193", 1: "UN9901",2: "UN0709",3: "UN1416",4:"UN2325"})
s[s < 0] = 0
X_param1_df_2024['UN3133'] = s

#calculating for year 2040 - UN
X_param1_df_2032 = np.array(X_param1_df_2024)
sess = tf.InteractiveSession()
best_beta_val_UN1 = np.array(best_beta_val_UN1, dtype=np.float64)
X_param1_2040 = tf.matmul(X_param1_df_2032[:,3:], best_beta_val_UN1, name="predictions")
s = X_param1_2040.eval()

#close the session to release resources
sess.close()

X_param1_df_2032 = pd.DataFrame(X_param1_df_2032)
X_param1_df_2032 = X_param1_df_2032.rename(index=str, columns={0: "UN9193", 1: "UN9901",2: "UN0709",3: "UN1416",4:"UN2325",5:"UN3133"})
s[s < 0] = 0
X_param1_df_2032['UN3941'] = s

X_param1_df_2032 = X_param1_df_2032.round(2)


#for STU calculation
X_param2=GHI_rdd4.map(lambda x: [x[2],x[6],x[10],x[14]]).filter(lambda x: x[0]!="stu9094")
X_param2=np.array(X_param2.collect())
print(X_param2)
#Beta value
best_beta_val_STU = RidgeRegression(X_param2, 1)

#calculating for year 2024 - STU
sess = tf.InteractiveSession()
best_beta_val_STU = np.array(best_beta_val_STU, dtype=np.float64)
X_param2_2024 = tf.matmul(X_param2[:,1:], best_beta_val_STU, name="predictions")
s = X_param2_2024.eval()

#close the session to release resources
sess.close()

X_param2_df = pd.DataFrame(X_param2)
X_param2_df = X_param2_df.rename(index=str, columns={0: "stu9094", 1: "stu9802",2: "stu0610",3: "stu1216"})
s[s < 0] = 0
X_param2_df['stu2226'] = s

#calculating 2032 - STU
X_param2_df_2024 = np.array(X_param2_df)
sess = tf.InteractiveSession()
best_beta_val_STU = np.array(best_beta_val_STU, dtype=np.float64)
X_param2_2032 = tf.matmul(X_param2_df_2024[:,2:], best_beta_val_STU, name="predictions")
s = X_param2_2032.eval()

#close the session to release resources
sess.close()

X_param2_df_2024 = pd.DataFrame(X_param2_df_2024)
X_param2_df_2024 = X_param2_df_2024.rename(index=str, columns={0: "stu9094", 1: "stu9802",2: "stu0610",3: "stu1216",4:"stu2226"})
s[s < 0] = 0
X_param2_df_2024['stu3034'] = s

#calculating 2040 - STU
X_param2_df_2032 = np.array(X_param2_df_2024)
sess = tf.InteractiveSession()
best_beta_val_STU = np.array(best_beta_val_STU, dtype=np.float64)
X_param2_2040 = tf.matmul(X_param2_df_2032[:,3:], best_beta_val_STU, name="predictions")
s = X_param2_2040.eval()

#close the session to release resources
sess.close()

X_param2_df_2032 = pd.DataFrame(X_param2_df_2032)
X_param2_df_2032 = X_param2_df_2032.rename(index=str, columns={0: "stu9094", 1: "stu9802",2: "stu0610",3: "stu1216",4:"stu2226",5:"stu3034"})
s[s < 0] = 0
X_param2_df_2032['stu3842'] = s

X_param2_df_2032 = X_param2_df_2032.round(2)


#calculating WAST
X_param3=GHI_rdd4.map(lambda x: [x[3],x[7],x[11],x[15]]).filter(lambda x: x[0]!="wast9094")
X_param3=np.array(X_param3.collect())
print(X_param3)
#calculating Beta value
best_beta_val_WAST = RidgeRegression(X_param3, 1)

#calculating for year 2024 - WAST
sess = tf.InteractiveSession()
best_beta_val_WAST = np.array(best_beta_val_WAST, dtype=np.float64)
X_param3_2024 = tf.matmul(X_param3[:,1:], best_beta_val_WAST, name="predictions")
s = X_param3_2024.eval()

#close the session to release resources
sess.close()

X_param3_df = pd.DataFrame(X_param3)
X_param3_df = X_param3_df.rename(index=str, columns={0: "wast9094", 1: "wast9802",2: "wast0610",3: "wast1216"})
s[s < 0] = 0
X_param3_df['wast2226'] = s

#calculating for year 2032 - WAST
X_param3_df_2024 = np.array(X_param3_df)
sess = tf.InteractiveSession()
best_beta_val_WAST = np.array(best_beta_val_WAST, dtype=np.float64)
X_param3_2032 = tf.matmul(X_param3_df_2024[:,2:], best_beta_val_WAST, name="predictions")
s = X_param3_2032.eval()

#close the session to release resources
sess.close()

X_param3_df_2024 = pd.DataFrame(X_param3_df_2024)
X_param3_df_2024 = X_param3_df_2024.rename(index=str, columns={0: "wast9094", 1: "wast9802",2: "wast0610",3: "wast1216",4: "wast2226"})
s[s < 0] = 0
X_param3_df_2024['wast3034'] = s


#calculating for year 2040 - WAST
X_param3_df_2032 = np.array(X_param3_df_2024)
sess = tf.InteractiveSession()
best_beta_val_WAST = np.array(best_beta_val_WAST, dtype=np.float64)
X_param3_2040 = tf.matmul(X_param3_df_2032[:,3:], best_beta_val_WAST, name="predictions")
s = X_param3_2040.eval()

#close the session to release resources
sess.close()

X_param3_df_2032 = pd.DataFrame(X_param3_df_2032)
X_param3_df_2032 = X_param3_df_2032.rename(index=str, columns={0: "wast9094", 1: "wast9802",2: "wast0610",3: "wast1216",4: "wast2226",5: "wast3034"})
s[s < 0] = 0
X_param3_df_2032['wast3842'] = s

X_param3_df_2032 = X_param3_df_2032.round(2)


#calculating for UM
X_param4=GHI_rdd4.map(lambda x: [x[4],x[8],x[12],x[16]]).filter(lambda x: x[0]!="UM1992")
X_param4=np.array(X_param4.collect())
print(X_param4)
#calculating Beta value
best_beta_val_UM = RidgeRegression(X_param4, 1)

#calculating UM value for year 2024
sess = tf.InteractiveSession()
best_beta_val_UM = np.array(best_beta_val_UM, dtype=np.float64)
X_param4_2024 = tf.matmul(X_param4[:,1:], best_beta_val_UM, name="predictions")
s = X_param4_2024.eval()

#close the session to release resources
sess.close()

X_param4_df = pd.DataFrame(X_param4)
X_param4_df = X_param4_df.rename(index=str, columns={0: "UM1992", 1: "UM2000",2: "UM2008",3: "UM2015"})
s[s < 0] = 0
X_param4_df['UM2024'] = s

#calculating UM value for year 2032
X_param4_df_2024 = np.array(X_param4_df)
sess = tf.InteractiveSession()
best_beta_val_UM = np.array(best_beta_val_UM, dtype=np.float64)
X_param4_2032 = tf.matmul(X_param4_df_2024[:,2:], best_beta_val_UM, name="predictions")
s = X_param4_2032.eval()

#close the session to release resources
sess.close()

X_param4_df_2024 = pd.DataFrame(X_param4_df_2024)
X_param4_df_2024 = X_param4_df_2024.rename(index=str, columns={0: "UM1992", 1: "UM2000",2: "UM2008",3: "UM2015",4: "UM2024"})
s[s < 0] = 0
X_param4_df_2024['UM2032'] = s

#calculating UM value for year 2040
X_param4_df_2032 = np.array(X_param4_df_2024)
sess = tf.InteractiveSession()
best_beta_val_UM = np.array(best_beta_val_UM, dtype=np.float64)
X_param4_2040 = tf.matmul(X_param4_df_2032[:,3:], best_beta_val_UM, name="predictions")
s = X_param4_2040.eval()

#close the session to release resources
sess.close()

X_param4_df_2032 = pd.DataFrame(X_param4_df_2032)
X_param4_df_2032 = X_param4_df_2032.rename(index=str, columns={0: "UM1992", 1: "UM2000",2: "UM2008",3: "UM2015",4: "UM2024",5: "UM2032"})
s[s < 0] = 0
X_param4_df_2032['UM2040'] = s

X_param4_df_2032 = X_param4_df_2032.round(2)


#creating lists form dataframes for each year
UM2024 = X_param4_df_2032['UM2024'].tolist()
UM2032 = X_param4_df_2032['UM2032'].tolist()
UM2040 = X_param4_df_2032['UM2040'].tolist()
wast2226 = X_param3_df_2032['wast2226'].tolist()
wast3034 = X_param3_df_2032['wast3034'].tolist()
wast3842 = X_param3_df_2032['wast3842'].tolist()
stu2226 = X_param2_df_2032['stu2226'].tolist()
stu3034 = X_param2_df_2032['stu3034'].tolist()
stu3842 = X_param2_df_2032['stu3842'].tolist()
UN2325 = X_param1_df_2032['UN2325'].tolist()
UN3133 = X_param1_df_2032['UN3133'].tolist()
UN3941 = X_param1_df_2032['UN3941'].tolist()
GHI2024_a = X_ghi_2032['GHI2024'].tolist()
GHI2032_a = X_ghi_2032['GHI2032'].tolist()
GHI2040_a = X_ghi_2032['GHI2040'].tolist()

#calculating GHI using UN, WAST, STU, UM
GHI2024_b = []
GHI2032_b = []
GHI2040_b = []
for i in range(0,len(UM2024)):
    x = (UN2325[i]/3) + (wast2226[i]/6) + (stu2226[i]/6) + (UM2024[i]/3)
    y = (UN3133[i]/3) + (wast3034[i]/6) + (stu3034[i]/6) + (UM2032[i]/3)
    z = (UN3941[i]/3) + (wast3842[i]/6) + (stu3842[i]/6) + (UM2040[i]/3)
    GHI2024_b.append(x)
    GHI2032_b.append(y)
    GHI2040_b.append(z)
    
GHI2024_b = [ round(elem, 2) for elem in GHI2024_b ]
GHI2032_b = [ round(elem, 2) for elem in GHI2032_b ]
GHI2040_b = [ round(elem, 2) for elem in GHI2040_b ]

#CALCULATING GHI BY TAKING MEAN OF TWO VALUES
GHI2024 = []
GHI2032 = []
GHI2040 = []
for i in range(0,len(GHI2024_a)):
    x = (GHI2024_a[i] + GHI2024_b[i])/2
    y = (GHI2032_a[i] + GHI2032_b[i])/2
    z = (GHI2040_a[i] + GHI2040_b[i])/2
    GHI2024.append(x)
    GHI2032.append(y)
    GHI2040.append(z)
    
GHI2024 = [ round(elem, 2) for elem in GHI2024 ]
GHI2032 = [ round(elem, 2) for elem in GHI2032 ]
GHI2040 = [ round(elem, 2) for elem in GHI2040 ]

#CONVERTING TO DATAFRAME
df1 = pd.DataFrame({'GHI2024': GHI2024,'GHI2032': GHI2032,'GHI2040': GHI2040})

#reading countries and countries codes
countries = pd.DataFrame({'COUNTRY':GHI_rdd2_unique_val})
ccodes = pd.read_csv("ccodes.csv")
result = pd.merge(countries, ccodes, on='COUNTRY')
result = result.drop(['GDP (BILLIONS)'],axis=1)

#visualization graphs for 2024 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = df1['GHI2024'],
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

#visualization graphs for 2032 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = df1['GHI2032'],
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
    title = 'Global Hunger Index - 2032',
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


#visualization graphs for 2040 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = df1['GHI2040'],
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
    title = 'Global Hunger Index - 2040',
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


#visualization graphs for 1992 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = X_ghi_2032['GHI1992'],
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
    title = 'Global Hunger Index - 1992',
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


#visualization graphs for 2000 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = X_ghi_2032['GHI2000'],
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
    title = 'Global Hunger Index - 2000',
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


#visualization graphs for 2008 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = X_ghi_2032['GHI2008'],
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
    title = 'Global Hunger Index - 2008',
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


#visualization graphs for 2015 year
data = [ dict(
        type = 'choropleth',
        locations = result['CODE'],
        z = X_ghi_2032['GHI2015'],
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
    title = 'Global Hunger Index - 2015',
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

