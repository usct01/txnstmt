# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:30:22 2018

@author: Deepak Mishra
"""
import pandas as pd

#READ THE DATA SHEET FROM EXCEL FILE
dt1=pd.read_excel("Data Science Assignment September 2018.xlsx",'Data')

#INSPECTING DATA
dt1.info()

dt1.drop(columns='YearMon',inplace=True)
dt1.head()

#SEE THE DESCRIPTIVE STATS
dt1.describe()

#RENAME FIELD Lead ID to Lead_ID
dt1.rename(columns={'Lead ID':'Lead_ID'},inplace=True)

#CREATE A FIELD FROM post_date to YYYY-MM
dt1['YearMon'] = dt1.post_date.dt.to_period('M')

#Exercises
#1. Count the number of merchants, bank accounts and tabulate:
Num_Merchants=dt1.Lead_ID.nunique()
print("Number of Unique Merchants: ",Num_Merchants)
#Number of Unique Merchants:   21

#(i) the number of bank accounts for each merchant
dt1.groupby('Lead_ID')['bank_account_id','account_number'].nunique()

#(ii) the number of months of each bank account for which data is available
#convert the result to dataframe for better looks and analysis
res_ii=dt1.groupby(['Lead_ID','bank_account_id']).YearMon.nunique().reset_index()
res_ii.head()
#326062            14045        2
#verify above result
dt1.loc[ dt1.bank_account_id==14045]

#(iii) the total number of credits (deposits), debits (withdrawals) 
#        and their averages per month for each bank account and each merchant
res_iii=dt1.groupby(['Lead_ID','bank_account_id','transaction_type']
            ).agg({'amount': ['size','mean']}).reset_index()
res_iii.head()
#verify results
res_iii.loc[res_iii.bank_account_id==12836]
dt1.loc[ dt1.bank_account_id==12836]

#(iv) the total dollar value of credits, debits and their averages per month 
#for each bank account and each merchant
res_iv=dt1.groupby(['Lead_ID','bank_account_id','transaction_type','YearMon']
            ).agg({'amount': ['size','sum','mean']}).reset_index()
res_iv.head()

#(v) Aggregate the answers to (iii) and (iv) 
#at the merchant level, industry level and bankid level
res_v=dt1.groupby(['Lead_ID','bankid','Industry']
            ).agg({'amount': ['size','sum','mean']}).reset_index()
res_v.head()

#(vi) Develop a slide visualizing the relationship between
# LeadID, bankid, bank_account_id and account_number 
dt1.groupby(['Lead_ID','bankid','bank_account_id','account_number']).size()

#Exercise 2. Consider Lead ID: 318465, bank_account_id: 13419  
#- plot the withdrawals, deposits and end-of-day daily balance 
#as a daily time series; do the same for Lead ID: 316728, bank_account_id : 13232.
#SELECT DESIRED ACCOUNT ID
dt1=dt1.loc[(dt1.Lead_ID==318465) &  (dt1.bank_account_id==13419)].reset_index()

#MULTIPLY THE DEBIT AMOUNTS BY -1
dt1.loc[dt1.transaction_type=='debit','amount']=-1*dt1.amount

#SORT THE DATA BY DATE AND TRANSACTION ORDER
dt1.sort_values(['Lead_ID','bankid','bank_account_id','post_date','trans_order'], inplace=True)

#SET THE OPENING BALANCE FOR EACH ACCOUNT
dt1['amount'].iloc[0]= dt1['running_balance'].values[0]

#COMPUTE EOD BALANCE
grpby=dt1.groupby(['Lead_ID','bankid','bank_account_id','post_date'])['amount'].sum().reset_index()
grpby['EOD_BAL']=grpby.amount.cumsum()

#PLOT THE EOD BALANCES
grpby.plot(x='post_date',y='EOD_BAL',kind='bar')

#CHECK THE DISTRIBUTION OF EOD BALANCE
grpby.EOD_BAL.hist()

#DESCRIPTIVE STATS OF EOD BALANCES
grpby.EOD_BAL.describe()

#DATA FOR WITHDRAWL AND DEPOSITS/ CREDIT AND DEBIT
#GET THE CREDIT AND DEBIT BY EOD
crdr=dt1.groupby(['Lead_ID','bankid','bank_account_id','post_date','transaction_type'])['amount'].sum().reset_index()
#GET THE CREDIT DEBIT AS SEPARATE COLUMNS FOR PLOTTING
pivt=(crdr.pivot_table(index=['Lead_ID','bankid','bank_account_id','post_date']
                      ,columns=  'transaction_type',values='amount' ).reset_index()
      )
pivt.fillna(value=0,inplace=True)
pivt.plot(x='post_date',y='credit',kind='bar')
pivt.plot(x='post_date',y='debit',kind='bar')


#3. For the merchants with the above Lead IDs, plot the withdrawals, deposits 
#and end-of-day daily balance as a daily time series 
#(aggregate over all their bank accounts)

#COMPUTE EOD BALANCE
#MULTIPLY THE DEBIT AMOUNTS BY -1
dt1.loc[dt1.transaction_type=='debit','amount']=-1*dt1.amount
# SELECT THE DATA FOR LEAD ID
dt11=dt1.loc[(dt1.Lead_ID==318465)].reset_index()
allacct = pd.DataFrame()
allpivt=pd.DataFrame()
#LOOP THOURGH ALL ACCOUNT ID OF A LEAD ID
for acct in dt11.bank_account_id.unique().tolist():
    #SELECT DESIRED ACCOUNT ID
    dt1=dt11.loc[(dt11.bank_account_id==acct)].reset_index()
    
    #SORT THE DATA BY DATE AND TRANSACTION ORDER
    dt1.sort_values(['Lead_ID','bankid','bank_account_id','post_date','trans_order'], inplace=True)
    
    #SET THE OPENING BALANCE FOR EACH ACCOUNT
    dt1['amount'].iloc[0]= dt1['running_balance'].values[0]
    
    #COMPUTE EOD BALANCE
    grpby=dt1.groupby(['Lead_ID','bankid','bank_account_id','post_date'])['amount'].sum().reset_index()
    grpby['EOD_BAL']=grpby.amount.cumsum()
    allacct = allacct.append(grpby)
    print(grpby.EOD_BAL.describe())
    #GET THE CREDIT AND DEBIT BY EOD
    crdr=dt1.groupby(['Lead_ID','bankid','bank_account_id','post_date','transaction_type'])['amount'].sum().reset_index()
    #GET THE CREDIT DEBIT AS SEPARATE COLUMNS FOR PLOTTING
    pivt=(crdr.pivot_table(index=['Lead_ID','bankid','bank_account_id','post_date']
                          ,columns=  'transaction_type',values='amount' ).reset_index()
          )
    pivt.fillna(value=0,inplace=True)
    allpivt=allpivt.append(pivt)
    print(pivt.describe())

#NOW CREATE EOD BALANCE AT MERCHANT LEVEL
#COMPUTE EOD BALANCE
mrhtEOD=allacct.groupby(['Lead_ID','post_date'])['amount'].sum().reset_index()
mrhtEOD['EOD_BAL']=mrhtEOD.amount.cumsum()

#PLOT THE EOD BALANCES
mrhtEOD.plot(x='post_date',y='EOD_BAL',kind='bar')

#CREDIT DEBIT AT MERCHANT LEVEL
crdrall=allpivt.groupby(['Lead_ID','post_date'])['credit','debit'].sum().reset_index()
pivt.plot(x='post_date',y='credit',kind='bar')
pivt.plot(x='post_date',y='debit',kind='bar')

#EXERCISE
#4. The cash flow of a business has many dimensions 
#[for example, top 5 deposits/Total deposits,  
# average # of days between two withdrawals, 
# coefficient of variation of the daily balance] 
#that are potentially relevant to a borrower’s ability to repay. 
#Can you obtain the value of these variables for each merchant?
# What other cash flow variables can you hypothesize that could be relevant 
# for predicting merchant default?

#TOP 5 DEPOSITS FOR EACH MERCHANT
deposits=dt1.loc[dt1.transaction_type=='credit']
topdep=deposits.groupby(['Lead_ID','post_date']).amount.sum().reset_index()
topdep.sort_values(by=['Lead_ID','amount'],ascending=[True,False],inplace=True)
Top5Dep=topdep.groupby('Lead_ID').head(5).reset_index()
print(Top5Dep)

#TOTAL DEPOSIT FOR EACH MERCHANT
TotalDep=topdep.groupby('Lead_ID').amount.sum().reset_index()
print(TotalDep)

#RATIO OF TOP 5 DEPOSITIS TO TOTAL DEPOSITS
Top5Dep=Top5Dep.groupby('Lead_ID').amount.sum().reset_index()
Top5Dep.rename(columns={'amount':'Top5Deposits'},inplace=True)
ratioDep=pd.merge(Top5Dep,TotalDep,on='Lead_ID')

ratioDep['Ratio']=ratioDep.Top5Deposits/ratioDep.amount
print(ratioDep)

# average # of days between two withdrawals, 
withdrawls=dt1.loc[dt1.transaction_type=='debit']

#df['diff'] = df.sort_values(['id','time']).groupby('id')['time'].diff()
withdrawls=deposits.groupby(['Lead_ID','post_date']).amount.sum().reset_index()
withdrawls['days_elapsed']=withdrawls.sort_values(['Lead_ID','post_date']).groupby('Lead_ID').post_date.diff().dt.days
withdrawls.describe()

avgdays=withdrawls.groupby('Lead_ID').days_elapsed.mean().reset_index()
print(avgdays)


# coefficient of variation of the daily balance
from scipy.stats import variation 
mrhtEOD.groupby(['Lead_ID']).EOD_BAL.apply(lambda x: variation(x)).reset_index()

# What other cash flow variables can you hypothesize that could be relevant 
# for predicting merchant default?
#Number of times running balance is negative in last 12 months
# Number of months when the average daily withdrawl is more than average daily deposits


#5. Develop a program by which clusters of similar transactions for a merchant
# can be identified from transaction descriptions and other information

dt11=dt1.loc[(dt1.Lead_ID==318465) & (dt1.transaction_type=='credit')].reset_index()
#GET THE TRANSACTION DESCRIPTION AS A LIST TO PROCESS
desc_trans=dt11.description.values.tolist()
#CHECK FEW VALUES
desc_trans[:3]

#WE WILL USE K MEANS CLUSTERING ON DESCRIPTIONS
#THERE ARE DIFFERENT WAYS WE CAN ANALYZE THIS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(desc_trans)

#####################################################
#FIND OUT OPTIMAL NUMBER OF CLUSTERS
#METHOD 1: Elbow curve based on SSE 
sse = {}
slh={}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    sse[k] = kmeans.inertia_ 
#    this is to find silhouette score
    label = kmeans.labels_
    slh[k] = silhouette_score(X, label, metric='euclidean')
#PLOT THE GRAPHS
import matplotlib.pyplot as plt
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#METHOD 2: Using  silhouette score: higher the better
plt.figure()
plt.plot(list(slh.keys()), list(slh.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
plt.show()


#####################################################
#GET num_clus FROM ABOVE 2 METHODS AND FIT THE FINAL CLSUTER
num_clus = 5
model = KMeans(n_clusters=num_clus, init='k-means++', max_iter=100, n_init=2)
model.fit(X)

print("Top terms per cluster:")
clus_cent = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clus):
    print("Cluster %d:" % i),
    for ind in clus_cent[i, :10]:
        print(' %s' % terms[ind]),
    print

# The model can be used to predict the cluster of new transaction
Y = vectorizer.transform(['ATM Deposit - BECU 317 NW GILMAN BLVD ISSAQUAH WAUS - Card Ending In 8648'])
prediction = model.predict(Y)
print(prediction)

