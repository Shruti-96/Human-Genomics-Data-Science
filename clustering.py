import numpy as np
import pandas as pd
import csv
#import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#data_array=pd.read_csv('abt_10.csv',skiprows=[0],header=None,delimiter = ' ')
#ind = data_array.ix[:,3]
#d = data_array.ix[:,4:]
#data = np.array(d)
#print(data)

colors = {'Khomani':'black','Karretjie':'red','Khwe':'gray','GuiGhanaKgal':'sandybrown','Juhoansi':'gold','Nama':'palegreen','Xun':'darkcyan','SEBantu':'royalblue','YRI':'darkgreen','CHB':'purple','CEU':'coral','indian':'navy','Test':'olive','JPT':'lawngreen','Biaka':'orange','KOR':'crimson','Bantu_N.E.':'c','KOR':'darkred','Mbuti':'hotpink','Bantu_S.E.':'pink','San':'yellow','EUR':'darkseagreen'}                         
data_array = pd.read_csv('abt_485.csv',skiprows=[0],header=None,delimiter = ' ')
reg = pd.read_csv('regions.csv',header=None,delimiter = ' ')
regions = {}
for i in range(len(reg)):
	regions[reg.iloc[i,0]] = reg.iloc[i,1]
ind = data_array.ix[:,0]
d = data_array.ix[:,1:]
data = np.array(d)

k = 4
num_clusters = k

f = open("hdbscan_results.csv", 'w')
fm = open("mds_results.csv","w")
writer = csv.writer(f,delimiter=" ")
writerm = csv.writer(fm,delimiter=" ")

clus = AgglomerativeClustering(n_clusters=k,linkage='average',affinity="precomputed") 
#clus = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')  # 4 clusters
#clus = hdbscan.HDBSCAN(min_cluster_size=2,metric = 'precomputed') 
#labels = clus.fit_predict(data)


#clus = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,dissimilarity="precomputed", n_jobs=1)
pos = clus.fit(data)
print(pos)
pos= StandardScaler().fit_transform(data)
print(pos)


for i in range(485):
	writerm.writerow([ind[i],pos[i,0],pos[i,1]])

fm.close()
fig = plt.figure(1)
#plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', lw=0, label='MDS')
for i in range(485):
	#plt.annotate(ind[i], (pos[i,0],pos[i,1]))
	place = regions[ind[i]]
	#print(place)
	plt.scatter(pos[i, 0], pos[i, 1], color=colors[place], lw=0, label='MDS')
	

plt.show()

my_cols = []
for i in range(len(set(labels))):
	my_cols.append("cluster"+str(i+1))

writer.writerow(my_cols)
#print(labels)
for i in range(len(labels)):
	row = [None]*len(set(labels))
	row[labels[i]] = ind[i]
	writer.writerow(row)

f.close()	
#print(labels)
print(silhouette_score(data,labels, metric='precomputed'))

'''
Khomani 
Khomani,KHO ----- Khomani in South Africa
Karretjie  -------- 
Karretjie,KHO
Khwe
Khwe,KHO
GuiGhanaKgal
GuiGhanaKgal,KHO
Juhoansi,KHO
Nama
Nama,KHO
Xun
Xun,KHO 
SEBantu ------ Africa Southeastern Bantu-speakers 
YRI --------Yoruba ,Yoruba in Ibadan, Nigeria (African AFR)
CEU ------- Utah residents (CEPH) with Northern and Western European ancestry  (European EUR)
CHB --------Han Chinese, Han Chinese in Beijing, China (East Asian EAS)
ABT Test
Biaka ------------- southern region of the Central African Republic ( tropical rain forests)
indian
JPT ---------Japanese in Tokyo, Japan (East Asian EAS)
KOR
Mbuti  ---------- Congo region of Africa
Bantu_N.E.  ---- Kenya in Subsaharian Africa
San,KHO    -------- Southern Africa (Botswana, Namibia, Angola, Zambia, Zimbabwe, Lesotho and South Africa)
Bantu_S.E. ---------South Africa in Subsaharian Africa
'''

