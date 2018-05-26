#!/usr/bin/env python
# -*- coding: utf-8 -*- 
#References of the code: https://developer.twitter.com/,http://adilmoujahid.com/posts/2014/07/twitter-analytics/
from __future__ import absolute_import, print_function
from bloomfilter import BloomFilter
from random import shuffle
from nltk.stem.snowball import SnowballStemmer
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from HTMLParser import HTMLParser
import json 
import os
import time
import sys
from itertools import chain
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
import datetime
import csv
import preprocessor

#Twitter Developer API authentication


# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="Pr5qbY4DjgNq7sbDgCo8sK3fu"
consumer_secret="VSJASScDTyKzddAQue2TdDB98dn4qCX22IgVTiuMBp1wPuPXeZ"
access_token="930208627708882944-EtqVQ8YYXINx9Z7mZoGvVkKvb6Rrxl4"
access_token_secret="IoP6Wb1628namo6s3AZWgsMtzpCgWpJjHmrj5zNSkLodj"

#Root word computation
#Tweets are in 'en' 
stemmer=SnowballStemmer("english")

#dictionary of the words made to filter the tweets from 
words=["hunger","craving","desire","famine","greed","longing","lust","starvation","yearning","ache","appetence",
"emptiness","esurience","famishment","gluttony","greediness","hungriness","mania","ravenous","vacancy","void","voracity",
"want","yen","Bankruptcy","debt","deficit","difficulty","hardship","lack","scarcity","shortage","underdevelopment","abjection",
"aridity","barrenness","beggary","dearth","deficiency","depletion","distress","emptiness","exiguity","impecuniousness",
"impoverishment","inadequacy","Indigence","insolvency","insufficiency","meagerness","necessity","pass","paucity",
"pauperism","Pennilessness","penury","pinch","poorness","privation","reduction","straits","vacancy","necessitousness","Drought","misery",
"destitution","paucity","malnourished","Bulimic","emaciated","thin","sickly","looking","without","appetite","thin",
"Poor","Destitute","impoverished","indigent","low","meager","needy","penniless","poverty-stricken","`poverty"
"Underprivileged","bankrupt","down-and-out","flat","insolvent","scanty","suffering","bad","off","Beggared","beggarly","dirt","poor","empty-handed",
"flat","broke","fortuneless","impecunious","moneyless","necessitous","pauperized","penurious","pinched","reduced","stone","strapped","unprosperous"]

#list of the countries from the which the tweets are matched for the location
countries=[["Afghanistan"], ["Albania"], ["Algeria"], ["American Samoa"],
["Andorra"], ["Angola"], ["Anguilla"], ["Antarctica"], ["Antigua and Barbuda"],
["Argentina"], ["Armenia"], ["Aruba"], ["Australia","Australian Capital Territory", "New South Wales", "Northern Territory", "Queensland", "South Australia", "Tasmania", "Victoria", "Western Australia"],
["Austria"], ["Azerbaijan"], ["Bahamas"], ["Bahrain"], ["Bangladesh"], ["Barbados"], 
["Belarus"], ["Belgium"], ["Belize"], ["Benin"], ["Bermuda"], ["Bhutan"], ["Bolivia"], 
["Bosnia and Herzegovina"],["Botswana"], ["Bouvet Island"],["Brazil","Acre", "Alagoas", "Amapa", "Amazonas", "Bahia", "Ceara", "Distrito Federal", "Espirito Santo", "Goias", "Maranhao", "Mato Grasso", "Mato Grosso do Sul", "Minas Gerais", "Parana", "Paraiba", "Pará", "Pernambuco", "Piaua", "Rio Grande do Norte", "Rio Grande do Sul", "Rio de Janeiro", "Rondônia", "Roraima", "Santa Catarina", "Sergipe", "Sao Paulo", "Tocantins"],
["British Indian Ocean Territory"], ["Brunei Darussalam"], ["Bulgaria"], ["Burkina Faso"], ["Burundi"],
["Cambodia"], ["Cameroon"], ["Canada","Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Northwest Territories", "Nova Scotia", "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon"], 
["Cape Verde"], ["Cayman Islands"], ["Central African Republic"], ["Chad"], ["Chile"], 
["China"], ["Christmas Island"], ["Cocos (Keeling Islands)"], ["Colombia"], ["Comoros"], ["Congo"], 
["Cook Islands"], ["Costa Rica"], ["Cote D'Ivoire (Ivory Coast)"], ["Croatia Hrvatska"], ["Cuba"], 
["Cyprus"], ["Czech Republic"], ["Denmark"], ["Djibouti"], ["Dominica"], ["Dominican Republic"], ["East Timor"],
["Ecuador"], ["Egypt"], ["El Salvador"], ["Equatorial Guinea"], ["Eritrea"], ["Estonia"], ["Ethiopia"], 
["Falkland Islands (Malvinas)"], ["Faroe Islands"], ["Fiji"], ["Finland"], ["France"], ["France, Metropolitan"],
["French Guiana"], ["French Polynesia"], ["French Southern Territories"], ["Gabon"], ["Gambia"], ["Georgia"], 
["Germany","Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"], 
["Ghana"], ["Gibraltar"], ["Greece"], ["Greenland"], ["Grenada"], ["Guadeloupe"], ["Guam"], ["Guatemala"], 
["Guinea"], ["Guinea-Bissau"], ["Guyana"], ["Haiti"], ["Heard and McDonald Islands"], ["Honduras"], 
["Hong Kong"], ["Hungary"], ["Iceland"], ["India","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar","Chandigarh","Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","Delhi","Puducherry"], 
["Indonesia"], ["Iran"], ["Iraq"], ["Ireland","Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin", "Galway", "Kerry", "Kildare", "Kilkenny", "Laois", "Leitrim", "Limerick", "Longford", "Louth", "Mayo", "Meath", "Monaghan", "Offaly", "Roscommon", "Sligo", "Tipperary", "Waterford", "Westmeath", "Wexford", "Wicklow"],
["Israel"], ["Italy"], ["Jamaica"], ["Japan"], ["Jordan"], ["Kazakhstan"], ["Kenya"], ["Kiribati"], ["Korea (North)"], ["Korea (South)"], ["Kuwait"], ["Kyrgyzstan"], ["Laos"], 
["Latvia"], ["Lebanon"], ["Lesotho"], ["Liberia"], ["Libya"], ["Liechtenstein"], ["Lithuania"], ["Luxembourg"], ["Macau"], ["Macedonia"], ["Madagascar"], ["Malawi"], ["Malaysia"], 
["Maldives"], ["Mali"], ["Malta"], ["Marshall Islands"], ["Martinique"], ["Mauritania"], ["Mauritius"], ["Mayotte"], ["Mexico","Aguascalientes", "Baja California Sur", "Baja California", "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Distrito Federal", "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Michoacán", "Morelos", "México", "Nayarit", "Nuevo León", "Oaxaca", "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas"],
["Micronesia"], ["Moldova"], ["Monaco"], ["Mongolia"], ["Montserrat"], ["Morocco"], ["Mozambique"], ["Myanmar"], ["Namibia"], ["Nauru"], ["Nepal"], ["Netherlands Antilles"], ["Netherlands","Drenthe", "Flevoland", "Friesland", "Gelderland", "Groningen", "Limburg", "Noord-Brabant", "Noord-Holland", "Overijssel", "Utrecht", "Zeeland", "Zuid-Holland"],
["New Caledonia"], ["New Zealand"], ["Nicaragua"], ["Niger"], ["Nigeria"], ["Niue"], ["Norfolk Island"], ["Northern Mariana Islands"], ["Norway"], ["Oman"], ["Pakistan"], ["Palau"], ["Panama"], ["Papua New Guinea"], ["Paraguay"], ["Peru"], ["Philippines"], ["Pitcairn"], ["Poland"], ["Portugal"], ["Puerto Rico"], ["Qatar"], ["Reunion"], ["Romania"], ["Russian Federation"], ["Rwanda"], ["S. Georgia and S. Sandwich Isls."], 
["Saint Kitts and Nevis"], ["Saint Lucia"], ["Saint Vincent and The Grenadines"], ["Samoa"], ["San Marino"], ["Sao Tome and Principe"], ["Saudi Arabia"], ["Senegal"], ["Seychelles"], ["Sierra Leone"], ["Singapore"], ["Slovak Republic"], ["Slovenia"], ["Solomon Islands"], ["Somalia"], ["South Africa"], ["Spain"], ["Sri Lanka"], ["St. Helena"], ["St. Pierre and Miquelon"], ["Sudan"], ["Suriname"], ["Svalbard and Jan Mayen Islands"], ["Swaziland"], 
["Sweden"], ["Switzerland"], ["Syria"], ["Taiwan"], ["Tajikistan"], ["Tanzania"], ["Thailand"], ["Togo"], ["Tokelau"], ["Tonga"], ["Trinidad and Tobago"], ["Tunisia"], ["Turkey"], ["Turkmenistan"], ["Turks and Caicos Islands"], ["Tuvalu"], ["US Minor Outlying Islands"], ["Uganda"], ["Ukraine"], ["United Arab Emirates"], ["United Kingdom"], 
["United States","Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District Of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"], ["Uruguay"], ["Uzbekistan"], ["Vanuatu"], ["Vatican City State (Holy See)"], ["Venezuela"], ["Viet Nam"], ["Virgin Islands (British)"], ["Virgin Islands (US)"], ["Wallis and Futuna Islands"],
["Western Sahara"], ["Yemen"], ["Yugoslavia"], ["Zaire"], ["Zambia"], ["Zimbabwe"]]

#Bloom Filter implementation in the code
# n is the number of words which are to be considered 
# p is the probability of false positive
n=40 
p=.05
bloomfilter=BloomFilter(n,p)
print("Bit array size:{}".format(bloomfilter.size))
print("False positive probability is:{}".format(bloomfilter.probability_false_positive))
print("Number of hash functions used in the bloomfilter:{}".format(bloomfilter.number_of_hf))

#Computation of the root words from the dictionary
#Adding the words to the bloom filter 
#This also ignores any non-ascii character from the tweet
root_words=[]
false_positive=[]
present_word=[]
not_present_word=[]
for w in words:
    bloomfilter.adding_item_bf(w)
    root_words.append(stemmer.stem(w).encode("ascii", "ignore"))


#The stream class to stream the tweets from the public API
#The tweets are cleaned 
#The text in the tweet is tokenize to get the tokens
#False positive is computed by checking it in the word
class StdOutListener(StreamListener):
    def on_data(self, data):
        cnt = 0
        root_tokens=[]
        try:
            data = json.loads(HTMLParser().unescape(data))
            cleaned_tweet=preprocessor.clean(data['text'].encode("ascii", "ignore"))
            date=data['created_at']
            tokens=nltk.word_tokenize(cleaned_tweet.encode("ascii", "ignore"))
            for w in tokens:
                if bloomfilter.checking_item_bf(w):
                    if w not in words:
                        false_positive.append(w)
                    else:
                        present_word.append(w)
                else:
                    not_present_word.append(w)                    
            #Appending the data into a csv file 
            with open("output123.csv", "a") as csv_file:    
                writer = csv.writer(csv_file, delimiter =",",quoting=csv.QUOTE_MINIMAL)
                for t in tokens:
                    root_tokens.append(stemmer.stem(t).encode("ascii", "ignore"))
                for rt in root_tokens:
                    if rt in root_words:
                        cnt = 1
                        if data['user']['location'] != None:
                            location = data['user']['location']
                        else:
                            location = ""
                if cnt == 1:
                    places = ""
                    for rt in tokens:  
                        if rt in [j for i in countries for j in i]:
                            places = rt
                    if location != "" or places != "":
                        writer.writerow([date,location,places,cleaned_tweet.encode("ascii", "ignore")])
     
            return True
        except BaseException as e:
            print('failed ondata',str(e))
            time.sleep(5)  

    def on_error(self, status):
        print("status: ",status)

if __name__ == '__main__':
    #listener for the stream class 
    listener = StdOutListener()
    #setting the auth handler for the consumer
    auth_handler = OAuthHandler(consumer_key, consumer_secret)
    #setting the access tokens 
    auth_handler.set_access_token(access_token, access_token_secret)
    #stream object called with the authentication
    stream_tweet = Stream(auth_handler, listener)
    #to filter the stream with words related to hunger poverty,poor,famine
    stream_tweet.filter(track=["hunger","malnourishment","poverty","poor","famine","underdevelopment"])


