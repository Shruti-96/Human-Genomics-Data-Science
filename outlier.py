import numpy as np
import pandas as pd
import csv
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import manifold
import matplotlib.pyplot as plt
#data_array=pd.read_csv('abt_10.csv',skiprows=[0],header=None,delimiter = ' ')
#ind = data_array.ix[:,3]
#d = data_array.ix[:,4:]
#data = np.array(d)
#print(data)
num=485
colors = {'Khomani':'black','Karretjie':'red','Khwe':'gray','GuiGhanaKgal':'sandybrown','Juhoansi':'gold','Nama':'palegreen','Xun':'darkcyan','SEBantu':'royalblue','YRI':'darkgreen','CHB':'purple','CEU':'coral','indian':'navy','Test':'olive','JPT':'lawngreen','Biaka':'orange','KOR':'crimson','Bantu_N.E.':'c','KOR':'darkred','Mbuti':'hotpink','Bantu_S.E.':'pink','San':'yellow','EUR':'darkseagreen'}                         
data_array = pd.read_csv('abt_485.csv',skiprows=[0],header=None,delimiter = ' ')
reg = pd.read_csv('regions.csv',header=None,delimiter = ' ')
regions = {}
for i in range(len(reg)):
	regions[reg.iloc[i,0]] = reg.iloc[i,1]
ind = data_array.ix[:,0]
d = data_array.ix[:,1:]
data = np.array(d)
X=[]
for i in range(num):
	for j in range(i+1,num):
		X.append(data[i][j])



n=data.shape[0]
trans_m=np.zeros((n,n))
for i in range(n):
	totSim=np.sum(data[i])
	for j in range(n):
		trans_m[i][j]=data[i][j]/totSim
#print(trans_m)
d=0.1
t=1
c_old=np.full((n,1),(1/n)) 
c_new=np.zeros((n,1))
#print(c_old)
tolerance=1e-10
error=d/n
while(error>tolerance):
	print("looping..."+str(t))
	c_new=(np.multiply(d,c_old))+(1-d)*(np.dot(np.transpose(trans_m),c_old))
	error=abs(np.amin(np.subtract(c_new,c_old)))
	#print(c_old.shape)
	c_old=c_new
	print("error"+str(error))
	#print(c_old.shape)
	t=t+1
	#break
y_out=[c_new[i][0] for i in range(n)]
X_arr=np.array(y_out)
mean_X=np.mean(X_arr)
std_X=np.std(X_arr)
print(std_X)
print(mean_X)

sorted_index=sorted(range(n), key=lambda ix: y_out[ix])
threshold=0.00195
#threshold=mean_X - std_X
print(threshold)
x=[i for i in range(1,n+1)]
y=[threshold for i in range(1,n+1)]
plt.plot(x,y,color='red',linestyle='--')
for i in range(485):
	place = regions[ind[i]]
	#print(place)
	plt.scatter(x[i], y_out[i], color=colors[place], lw=0,marker='o')
#plt.plot(x,y_out,color='blue',linestyle='',marker='o')
plt.show()












