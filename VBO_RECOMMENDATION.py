import pandas as pd
import numpy as np
import math
import scipy.stats as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats._stats_py import ttest_ind
import matplotlib as mt
import statsmodels.stats.api as sms
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
# Recomendation Systems
from mlxtend.frequent_patterns import apriori, association_rules
# Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

## RECOMMENDATION SYSTEMS

# Basicaly, using special methods to make recommendation to customer for a service or product.
# 90 larda dijital kitaplık adıyla ilk defa bahsedilmiş/ kullanılmıştır.
# İçeriğin oldukça bol, kullanıcının ilgi alanının daha küçük olduğu durumda, o bol içerik FİLTRELENEBİLSİN ki
# kullanıcıya anlamlı bir şekilde ulaştırılabilsin.
# House of Cards Netfilixin ilk dizisidir. 2013 yılında Netflix bir yarışma düzenlemiş ve tavsiye sistemini %10
# geliştirebilene 1 milyon dolarlık ödül vermiştir. (Robert Bell)
# 500 bin kadar kullanıcı, 200 bin kadar filme yaklaşık 100 milyon puanlama yapmıştır. Amaç kullanıcı-film puanı çiftleri
# üzerinden bir öğrenme algoritması geliştirerek, bilinmeyen çiftleri tahmin etmek ve kullanıcıya tutarlı tahmin öneride
# bulunmaktır.
## Subjects
# Simple Reccomender System (It can be done with simple teknikler and business knowledge)
# Association Rule Learning (Birliktelik Kuralı Yöntemi / Sepet Analizi)(Çok sık birlikte satın alınan ürünlerin olasılıklarını
#   çıkarır ve öneriler yapmayı sağlar.)
# Content Based Filtering (meta etiketleri ve açıklamaları kullanarak tavsiye verme)
# Collaborative Filtering (Topluluklar üzerinden tavsiye verme)(User-Item Based ve Model Based are in this title)

# Association Rule Learning (Birliktelik Kuralı Öğrenimi)
# Used for finding patterns, relations in data, based machine learning (10 years ago its called 'data mining')
# Now it can be called as a AI method.
# Wallmart finds that there is a relation between baby towels and beer :)

##Apriori Algorithm
# Used for calculating Association in data.
# There is 3 essential parameters;
# Support = freq( x and y together)/(all transaction)
# Confidence = Freq(X,Y)/Freq(X) : Probability of Y when X purchased (ekmek satın alındığında süt satın alınma olasılığı)
# Lift = Support(X,Y)/(Support(X) * Support(Y)) = When X purchased, product Y's probability of purchaseing increase by LIFT
## How Apirori Algorithm works? (Exercise on EXCEL)
# Apirori algorithm, calculates possible product couples and makes eliminations in every iteration
# acording to a "support" threshold value that determind at the begining.
# As a result, creates a final table.
# 1) Calculate Supports for every product.
# 2) Eliminate products lower then threshold.
# 3) Create possible couples with remane products and THEIR support values.
# 4) Eliminate products lower then threshold value again.
# 5) Number 3 again.
# 6) Repeat the process until 1 remain.
# 7) Create final table with every product couples that survives every iteration.
# 8) Calculate Freq, Support, Confidence and Lift values for every product and couples
#    from Step-7
## Interpration of Values
# Support: Egg and Tea observed together %40.
# Confidence: %67 of customers that purchased Egg are buys Tea too.
# Lift: Transactions within Egg, Tea purchases increase 1.11 times.

## Association Rule Based Recommender System
# Here is the thing about this exercise, normally data sets online are
# suitable for Apirori algorithm but we reformat the data ourselfs.
# 1) Data Pre-Processing
# 2) Preparing ARL data structure. (Invoice-Prodduct Matrix)
# 3) Birliktelik Kurallarının Çıkarılması
# 4) Writing Script
# 5) Making Suggestions to Check-Out Status Customers

df_ = pd.read_excel('/Users/buraksayilar/Desktop/recommender_systems/datasets/online_retail_II.xlsx',
                    sheet_name='Year 2010-2011')
# If problem accures while uploading data set; pip install openpyxl
# ,engine='openpyxl'
df = df_.copy()
df.head()
# Real challange in this project is transforming data set. Rather then functions
# or interpreting results, like many projects.
df.describe().T
df.isnull().sum()
# I should get rid of NAN values, Canceled oreders and quantity,price values
# equal 0.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[dataframe['Quantity'] > 0]
    dataframe = dataframe[dataframe['Price'] > 0]
    return dataframe
df = retail_data_prep(df)

# My priority is get rid of the outliers in Quantity, but incase of
# another use of data, I will clear outliers in Price too.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (interquantile_range*1.5)
    low_limit = quartile1 - (interquantile_range*1.5)
    return low_limit, up_limit
def replace_with_thersholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[dataframe['Quantity'] > 0]
    dataframe = dataframe[dataframe['Price'] > 0]
    replace_with_thersholds(dataframe, 'Quantity')
    replace_with_thersholds(dataframe, 'Price')
    return dataframe
df["Invoice"] = df["Invoice"].astype(str)
df = retail_data_prep(df)
df.describe().T


## Prepearing ARL Data Structure
# We are gonna reduce our dataset to a one country; France.
df_fr = df[df['Country'] == 'France']
# As a business case, for instance our company will expand to the Germany market.
# But we dont have any data about Germany market. So we can examin another country that
# relatively similar market; like France for strategy.
df_fr.groupby(['Invoice', 'Description']).agg({'Quantity':'sum'}).head(20)
df_fr.groupby(['Invoice', 'Description']).agg({'Quantity':'sum'}).unstack().iloc[0:5, 0:5]
# We pivot the table (means; making Descriptions as variable names). Then with iloc,
# we make index based choice and say 'get me 5 rows and 5 columns'.
# Okay, but there is a problem, we want this table with 1 and 0. For this
# first we fill NaN values with 0, with fillna(0) method.
df_fr.groupby(['Invoice', 'Description']). \
    agg({'Quantity':'sum'}).\
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).\
    iloc[0:5, 0:5]
# applymap is apply method's brother. apply func. takes row or column info and
# travels between this rows or columns without loops.
# applymap func. travels around all observations.

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode']). \
                   agg({'Quantity':'sum'}).\
                   unstack(). \
                   fillna(0). \
                   applymap(lambda x: 1 if x > 0 else 0)
    else:
        return df_fr.groupby(['Invoice', 'Description']). \
                   agg({'Quantity': 'sum'}). \
                   unstack(). \
                   fillna(0). \
                   applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)
# For checking products descriptions esayly
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe['StockCode'] == stock_code][['Description']].values[0].tolist()
    print(product_name)


## Birliktelik Kurallarının Çıkarılması
# With Apirori func. we will calculate all possible product pairs and their
# support values.
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.sort_values('support', ascending=False)
# Now we can observe these possible pairs and theri probability to seen together.
# Couples that have possibilty lower then 0.01 are not listed!
# Now its time to calculate association_rules.
rules = association_rules(frequent_itemsets,
                          metric='support',
                          min_threshold=0.01)
# We can observe more comprehensive way with other paremeters like lift,
# confidence.
# antecedents: önceki ürün
# consequents: ikinci ürün
# antecedent support: ilk ürünün tek başına gözlenme olasılığı
# consequent support: ikinci ürünün tek başına gözlenme olasılığı
# support: iki ürünün birlikte görünme olasılığı
# confidence: X ürünü alındığında Y'nin alınma olasılığı.
# lift: X ürünü satın alındığında Y ürününün satın alınma olasılığı 'lift' kat artar.
# leverage: lifte benzer bir değerdir ancak support' u yüksek olan değerlere öncelik verme eğilimindedir. Lift
# değeri daha az sıklıkta olmasına rağmen bazı ilişkileri yakalayabilmektedir. Bizim için daha yansız bir metriktir.
# conviction: Y ürünü olmadan X ürününün beklenen frekansıdır.
# Biz daha çok lift support ve confidence değerlierini kullanacağız

rules[(rules['confidence'] > 1.5) & (rules['support'] > 1.5)].sort_values('lift', ascending=False)
rules[rules['confidence'] > 1]