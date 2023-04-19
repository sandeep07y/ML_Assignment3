import numpy as np
import pandas as pd
#from sklearn.cluster import KMeans
import scipy 
import sklearn
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from scipy import spatial
dataset = pd.read_csv('data.csv')
labels = pd.read_csv('label.csv',names=['label'],header=None)
dataset.count()
from sklearn.model_selection import train_test_split
train_values, test_values = train_test_split( dataset, test_size=0.08, random_state=50)
train_label_val, test_label_val = train_test_split( labels, test_size=0.08, random_state=50)
def calculate_SSE(centroid_value_dict, centroid_dict,data):
    error = 0
    for i in centroid_dict:
        SSE_values = 0
        for j in centroid_dict[i]:
            dp = list(data.iloc[int(j)])
            for a,b in zip(centroid_value_dict[i],dp):
                SSE_values =SSE_values+ (a-b)**2
        error= error+ SSE_values
    return error   
    
def Initialize_Centroids(data,K):
    p = data.shape[0]
    centroid_value_dict={}
    for i in range(K):
        r = np.random.randint(0, p-1)
        centroid_value_dict[i] = data.iloc[r]
    return centroid_value_dict

def jaccard_similarity(centroid, dp):
    top = len(list(set(centroid).intersection(dp)))
    bottom = (len(set(centroid)) + len(set(dp))) - top
    return float(top) / bottom

def train_Kmeans(data,K,max_iter=20,mode=1,tol=10):
    centroid_value_dict = Initialize_Centroids(data,K)
    count = 0
    centroid_dict = {}
    flag = False
    while((count<max_iter) and not flag):
            
        for i in list(centroid_value_dict.keys()):
            centroid_dict[i]=[]
        for i in range(data.shape[0]):
            x = data.iloc[i]
            if mode==1 :
                distance_measure = [np.linalg.norm(x-centroid_value_dict[j])  for j in centroid_value_dict]
                idx = np.argmin(distance_measure)
                centroid_dict[idx].append(i)
            elif mode==2 :
                distance_measure = [jaccard_similarity(list(x),centroid_value_dict[j]) for j in centroid_value_dict]
                idx = np.argmax(distance_measure)
                centroid_dict[idx].append(i)
            elif mode==3 :
                distance_measure = [1-scipy.spatial.distance.cosine(x,list(centroid_value_dict[j]))  for j in centroid_value_dict]
                idx = np.argmax(distance_measure)
                centroid_dict[idx].append(i)
                
            prev_centroids=dict(centroid_value_dict)
        for i in centroid_dict:
            if len(centroid_dict[i]):
                dps_centroid = centroid_dict[i]
                centroid_value_dict[i] = np.average(data.iloc[dps_centroid],axis=0)
        current_tol=-1
        for i in centroid_value_dict:
            prev_centroid_point = prev_centroids[i]
            new_centroid_point = centroid_value_dict[i]
            change = np.sum(np.absolute(new_centroid_point-prev_centroid_point))
            current_tol = max(change, current_tol)
                
        print("Iteration ",count,": ",current_tol)
            
        count+=1
        if (current_tol<10):
            flag = True
            break
    return centroid_value_dict,centroid_dict
def predict_cluster_labels(C, S, labels):
    cluster_labels = np.zeros(10,dtype=int)
    for c in C:
        labels_of_points = []
        for point in S[c]:
            labels_of_points.extend(labels.iloc[point])
        counter = Counter(labels_of_points)
        try:
            cluster_labels[c] = max(counter, key=counter.get)
        except:
            cluster_labels[c] = np.random.randint(0,9)
    return cluster_labels
def accuracy(centroids, centroid_Labels, test_data, true_labels, mode=1):
    y_true = list(true_labels['label']);
    y_values = []
    for index in range(test_data.shape[0]):
        featureset = test_data.iloc[index]
        if mode==1:
            distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            y_values.append(centroid_Labels[classification])
        elif mode==2:
            similarity = [jaccard_similarity(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_values.append(centroid_Labels[classification]) 
        elif mode==3:
            similarity = [1 - spatial.distance.cosine(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_values.append(centroid_Labels[classification])
    denominator = test_data.shape[0]
    correctly_classified = 0
    for i in range(0,len(y_values)):
        if y_true[i] == y_values[i]:
            correctly_classified += 1
    accuracy = correctly_classified/denominator
    return accuracy
centroids1,clusters1 = train_Kmeans(dataset,10, max_iter=100,mode=1)
Euclidean_SSE =calculate_SSE(centroids1,clusters1,dataset)
print("Euclidean SSE:",Euclidean_SSE)
cluster_labels_euc = predict_cluster_labels(centroids1,clusters1,labels)
cluster_labels_euc
Accuracy_Euclidean = accuracy(centroids1, cluster_labels_euc,test_values,test_label_val)
Accuracy_Euclidean
centroids2,clusters2 =train_Kmeans(dataset,10, max_iter=100,mode=2)
Jaccard_SSE =calculate_SSE(centroids2,clusters2,dataset)
print("Jacard SSE:",Jaccard_SSE)
cluster_labels_jac = predict_cluster_labels(centroids2,clusters2,labels)
cluster_labels_jac
Accuracy_Jaccard = accuracy(centroids2, cluster_labels_jac,test_values,test_label_val,mode=2)
Accuracy_Jaccard
centroids3,clusters3 =train_Kmeans(dataset,10, max_iter = 100,mode=3)
Cosine_SSE = calculate_SSE(centroids3,clusters3,dataset)
cluster_labels_cos = predict_cluster_labels(centroids3,clusters3,labels)
cluster_labels_cos
Accuracy_Cosine = accuracy(centroids3, cluster_labels_cos,test_values,test_label_val,mode=3)
print("Euclidean accuracy:",Accuracy_Euclidean)
print("Jacard accuracy:",Accuracy_Jaccard)
print("Cosine accuracy :",Accuracy_Cosine)
print("Euclidean SSE value:",Euclidean_SSE)
print("Jacard SSE value:",Jaccard_SSE)
print("Cosine SSE value:",Cosine_SSE)
