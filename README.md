# Zero-Hunger-Sustainable-Development-Goal-2

Sustainable Development Goal 2 - Zero Hunger
Big Data Project (under prof. Andrew Schwartz)

Languages, libraries & frameworks used: Python, Spark, TensorFlow, Pandas

Responsibilities:
1. Scraped data from twitter and applied Bloom filter with 5% error probability while streaming.
2. Performed data cleaning and normalization on 5GB of data in Spark RDDs.
3. Applied Ridge regression using TensorFlow to predict future trends, and visualized data using Pandas DataFrames and choropleth maps.


Description:
Following is the description of all the files:

	•	Scrapdata.py: file to stream data from Twitter API 
	•	countriesMap.py: file to obtain the count of countries both inside and outside the tweet 
	•	Bloomfilter.py: file storing bloom filter code for the scrapdata.py
	•	GHI.py: Global Hunger Index Dataset to predict and plot the GHI for different countries (used for finding correlation with WDI)
	•	WDI.py: World Development Indicator Dataset to predict and plot the WDI for different countries (used for finding correlation with GHI)
	•	FAO.py: Food and Agriculture Organization Dataset to analyze and plot total food production for each country
	•	GST.py: Global Surface Temperature Dataset to analyze the climate trends and see how they affect the food production of each country
	•	Report.pdf is the project report. 
