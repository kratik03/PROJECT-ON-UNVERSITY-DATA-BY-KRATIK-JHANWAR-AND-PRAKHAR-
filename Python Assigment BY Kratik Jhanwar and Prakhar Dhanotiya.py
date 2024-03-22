#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Python Assigment BY Kratik Jhanwar and Prakhar Dhanotiya on Unversity Result")


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sls
Final=pd.read_csv("FINAL.csv")
Mid_term=pd.read_csv("mid term.csv")
Final= Final.fillna(0)


# In[7]:


roll=[x for x in Final.iloc[:,0]]
fp=[x for x in Final.iloc[:,4]] 
qp=[x for x in Mid_term.iloc[:,4]] 
ap=[x for x in Mid_term.iloc[:,-3]] 
mp=[x for x in Mid_term.iloc[:,-2]] 
sp=[x for x in Mid_term.iloc[:,-1]] 
tp=[x for x in Final.iloc[:,3]] 
def percentile_rank(dataset, value):
    less_than_value = sum(1 for x in dataset if x < value)
    total_values = len(dataset)
    percentile = (less_than_value / total_values) * 100
    return percentile
list_percentile=[]
for value in fp:
    
    percentile = format(percentile_rank(fp, value),'.2f')
    list_percentile.append(percentile)
Final['Percentile']=list_percentile    
print(Final.head(5))

print(Mid_term.head(5))


# In[8]:


F=int(input('Enter Roll Number of Student whose mid term Detail you want to get',))
for i in Mid_term.iloc[:,0]:
      if i==F:
            
            E=Mid_term[Mid_term['Roll_No.'] == F]
       
    
print(E)


# In[12]:


AA=int(input('Enter Roll Number of Student whose Final Results you want to get',))
for i in Final.iloc[:,0]:
      if i==AA:
            BB=Final[Final['Roll_No.'] == AA]
       
print(BB)


# In[13]:


print(BB.iloc[0,1],"Marks Compostion")
y=AA+1
Pie=[ap[y],qp[y],mp[y],tp[y]]
plt.pie(Pie,labels=["Assig","Quiz","Mid-term",'Terminal'],autopct='%1.1f%%')
plt.axis('equal')
plt.show()


# In[14]:


print(BB.iloc[0,1])
LIST=[sp[y],tp[y]]
LIST1=["Sessional","Terminal"]

plt.bar(LIST1,LIST)


for i, value in enumerate(LIST):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.xlabel('NAME OF EXAM')
plt.ylabel('MARK OF STUDENT')
plt.title('CAMPARSION BETWEEN SESSIONAL AND TERMIAL')
plt.show()
if sp[y]>tp[y]:
    print("Remark",'Need Improvement')
else:
    print('Remark', "Keep it up")


# In[15]:


i=0
A=0
B=0
C=0
F=0
D=0
for i in Final.iloc[:,-3]:
    if i=="A":
        A=A+1
    elif i=="B+":
        B=B+1
    elif i=="C":
        C=C+1
    elif i=="F":
        F=F+1
    elif i=="B":
        D=D+1
X = ["A","B+","B","C","F"]
Y=[A,B,D,C,F]
plt.bar(X,Y)
sls.barplot(X,Y)

for i, value in enumerate(Y):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.title('Grade Counts')
plt.show()


# In[16]:


import heapq
fp=[x for x in Final.iloc[:,4]] 
qp=[x for x in Mid_term.iloc[:,4]] 
ap=[x for x in Mid_term.iloc[:,-3]] 
mp=[x for x in Mid_term.iloc[:,-2]] 
sp=[x for x in Mid_term.iloc[:,-1]] 
tp=[x for x in Final.iloc[:,3]] 
print(max(fp))
top_five_preformer=(max(fp))
Top_three_Preformer=[w for w in heapq.nlargest(3, fp)]
print(Top_three_Preformer)
First_Rank = Final[Final['Total(150)'] == Top_three_Preformer[0]]
Second_Rank= Final[Final['Total(150)'] == Top_three_Preformer[1]]
Third_Rank = Final[Final['Total(150)'] == Top_three_Preformer[2]]
print('Roll Number of First_Ranker ',First_Rank.iloc[0,0])
print('Name of First_Ranker',First_Rank.iloc[0,1])
print('Roll Number of Second_Ranker',Second_Rank.iloc[0,0])
print('Name of Second_Ranker',Second_Rank.iloc[0,1])
print('Roll Number of Third_Ranker',Third_Rank.iloc[0,0])
print('Name of Third_Ranker',Third_Rank.iloc[0,1])
Top_three_tPreformer=[w for w in heapq.nlargest(3, tp)]
print(Top_three_tPreformer)
First_tRank = Final[Final['Terminal(75)'] == Top_three_tPreformer[0]]
Second_tRank= Final[Final['Terminal(75)'] == Top_three_tPreformer[1] ]
Third_tRank = Final[Final['Terminal(75)'] == Top_three_tPreformer[2]]
print('Roll Number of First_Ranker ',First_tRank.iloc[0,0])
print('Name of First_Ranker',First_tRank.iloc[0,1])
print('Roll Number of Second_Ranker',Second_tRank.iloc[0,0])
print('Name of Second_Ranker',Second_tRank.iloc[0,1])
print('Roll Number of Third_Ranker',Third_tRank.iloc[0,0])
print('Name of Third_Ranker',Third_tRank.iloc[0,1])
Top_three_sPreformer=[w for w in heapq.nlargest(3, sp)]
print(Top_three_sPreformer)
First_sRank = Final[Final['Sessional(75)'] == Top_three_sPreformer[0]]
Second_sRank= Final[Final['Sessional(75)'] == Top_three_sPreformer[1] ]
Third_sRank = Final[Final['Sessional(75)'] == Top_three_sPreformer[2]]#Sessional(75)
print('Roll Number of First_Ranker ',First_sRank.iloc[0,0])
print('Name of First_Ranker',First_sRank.iloc[0,1])
print('Roll Number of Second_Ranker',Second_sRank.iloc[0,0])
print('Name of Second_Ranker',Second_sRank.iloc[0,1])
print('Roll Number of Third_Ranker',Third_sRank.iloc[0,0])
print('Name of Third_Ranker',Third_sRank.iloc[0,1])
Top_three_mPreformer=[w for w in heapq.nlargest(7, mp)]
print(Top_three_mPreformer)
First_mRank = Mid_term[Mid_term['Mid_term(45)'] == Top_three_mPreformer[0]]
Second_mRank= Mid_term[Mid_term['Mid_term(45)'] == Top_three_mPreformer[1]]
Third_mRank = Mid_term[Mid_term['Mid_term(45)'] == Top_three_mPreformer[2]]
print('Roll Number of First_Ranker ',First_mRank.iloc[0,0])
print('Name of First_Ranker',First_mRank.iloc[0,1])
print('Roll Number of Second_Ranker',Second_mRank.iloc[0,0])
print('Name of Second_Ranker',Second_mRank.iloc[0,1])
print('Roll Number of Third_Ranker',Third_mRank.iloc[0,0])
print('Name of Third_Ranker',Third_mRank.iloc[0,1])
Avg_t_mean=np.mean(Final.iloc[:,4])
Avg_T_mean=np.mean(Final.iloc[:,3])
Avg_S_mean=np.mean(Final.iloc[:,2])
f_median=np.median(Final.iloc[:,4])
T_median=np.median(Final.iloc[:,3])
S_median=np.median(Final.iloc[:,2])
std_dev_f = np.std(Final.iloc[:,4])
print(std_dev_f)
std_dev_t = np.std(Final.iloc[:,3])
print(std_dev_t)
std_dev_s = np.std(Final.iloc[:,2])
print(std_dev_s)


# In[17]:


from tabulate import tabulate
data = [
    
    ["Rank 1",First_sRank.iloc[0,1],First_tRank.iloc[0,1],First_Rank.iloc[0,1]],
    ["Rank 2",Second_sRank.iloc[0,1],Second_tRank.iloc[1,1],Second_Rank.iloc[0,1]],
    ['Rank 3',Third_sRank.iloc[0,1],Third_tRank.iloc[0,1],Third_Rank.iloc[0,1]],
    ["AVG Marks",format(Avg_S_mean,'.2f'),format(Avg_T_mean,'.2f'),format(Avg_t_mean,'.2f')],
    ["Median Marks",S_median,T_median,f_median],
    ['standard deviation',std_dev_s,std_dev_t,std_dev_f]
    
]


table = tabulate(data,headers=["","Sessional","Terminal", "Final"])

print(table)


# In[9]:


import scipy.stats as stats
MId_term_scaled_score=[format(x*75/45,".2f") for x in mp]


group1 = MId_term_scaled_score
group2 = tp



f_statistic, p_value = stats.f_oneway(group1, group2)

print("F-Statistic:", f_statistic, end=' ')
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject null hypothesis: There is a significant difference between marks of Mid term and  End sem means.")
else:
    print("Fail to reject null hypothesis: There is no significant difference between marks of Mid term and  End sem means.")


# In[ ]:




