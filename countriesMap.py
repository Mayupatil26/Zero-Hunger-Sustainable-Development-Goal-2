#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from operator import itemgetter
from pyspark import SparkConf, SparkContext
from collections import Counter
import matplotlib.pyplot as plt
import string

#rdd to fetch data from the csv file of tweets
#Setting the spark context
sc = SparkContext("local", "countriesCountMap")
#rdd reading the csv file and then splitting the lines by comma
rdd=sc.textFile("output123.csv").map(lambda line:line.split(",")).filter(lambda line:len(line)==4)

#Calculation of the location outside the tweet 
location_outside=rdd.filter(lambda x: x[1] is not None).filter(lambda x: x[1] != "").map(lambda x:(x[1].encode("ascii", "ignore").translate(None, string.punctuation))).collect()
locationfreq = [(w,location_outside.count(w)) for w in location_outside]
location_out=list(set(locationfreq))
location_out_dup=sorted(locationfreq,key=lambda x:x[1])
loc_out=[]
for items in location_out_dup:
	if items not in loc_out:
		loc_out.append(items)

print(loc_out)

#Calculation of the tweet inside the tweet 
print("Location inside the tweet")
location_inside=rdd.filter(lambda x:x[2] is not None).filter(lambda x:x[2]!="").map(lambda x:(x[2].encode("ascii", "ignore").translate(None, string.punctuation))).collect()
location_freq=[(w,location_inside.count(w)) for w in location_inside]
location_in_dup=sorted(location_freq,key=lambda x:x[1])
loc_in=[]
for x in location_in_dup:
	if x not in loc_in:
		loc_in.append(x)
print(loc_in)





