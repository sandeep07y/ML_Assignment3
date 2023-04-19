import sys
!{sys.executable} -m pip install scikit-surprise
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from surprise import KNNBasic, SVD, Reader, accuracy, Dataset
from surprise.model_selection import cross_validate, train_test_split
%matplotlib inline
from google.colab import files
uploaded = files.upload()
import io
rating = pd.read_csv(io.BytesIO(uploaded['ratings_small.csv']))
rating
rating.info
rating.describe
rating['userId'].value_counts()
rating['movieId'].value_counts()
reader = Reader()
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
probmf_svd = SVD(biased = False)
cv_probmf = cross_validate(probmf_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
cv_probmf
print('MAE and PMF average for Collaborative Filtering is  ', cv_probmf['test_mae'].mean())
print('RMSE and PMF average for Collaborative Filtering is ', cv_probmf['test_rmse'].mean())
sim_options = {'user_based': True}
user_colf = KNNBasic(sim_options=sim_options)
cv_ub = cross_validate(user_colf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
print('MAE Average for Userbased collaborative filtering is  ', cv_ub['test_mae'].mean())
print('RMSE Average for Userbased collaborative filtering is  ', cv_ub['test_rmse'].mean())
sim_options = {'user_based': False}
item_based_colf = KNNBasic(sim_options=sim_options)
cv_itb = cross_validate(item_based_colf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
print('MAE Average for Itembased collaborative filtering is ', cv_itb['test_mae'].mean())
print('RMSE Average for Itembased collaborative filtering is ', cv_itb['test_rmse'].mean())
sim_options = {'name':'cosine', 'user_based': True}
usb_cosine = KNNBasic(sim_options=sim_options);
cv_usb_cos = cross_validate(usb_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
sim_options = {'name':'msd', 'user_based': True}
usb_msd = KNNBasic(sim_options=sim_options);
cv_usb_msd = cross_validate(usb_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
sim_options = {'name':'pearson', 'user_based': True}
usb_pearson = KNNBasic(sim_options=sim_options);
cv_usb_pearson = cross_validate(usb_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
x = [0,1,2]
y_mae = [cv_usb_cos['test_mae'].mean(),cv_usb_msd['test_mae'].mean(),cv_usb_pearson['test_mae'].mean()]
#y_rmse = [cv_usb_cos['test_rmse'].mean(),cv_usb_msd['test_rmse'].mean(),cv_usb_pearson['test_rmse'].mean()]
plt.plot(x, y_mae)
#plt.plot(x, y_rmse)
plt.title('UserBased Collaborative Filtering for 5 fold cross validation')
plt.legend(['MAE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity measure')
plt.ylabel('Average MAE')
plt.show()
x = [0,1,2]
y_rmse = [cv_usb_cos['test_rmse'].mean(),cv_usb_msd['test_rmse'].mean(),cv_usb_pearson['test_rmse'].mean()]
plt.plot(x, y_rmse)
plt.title('UserBased Collaborative Filtering for 5 fold cross validation')
plt.legend(['RMSE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity measure')
plt.ylabel('Average RMSE')
plt.show()
sim_options = {'name':'cosine', 'user_based': False}
itb_cos = KNNBasic(sim_options=sim_options);
cv_itb_cos = cross_validate(itb_cos, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
sim_options = {'name':'msd', 'user_based': False}
itb_msd = KNNBasic(sim_options=sim_options);
cv_itb_msd = cross_validate(itb_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
sim_options = {'name':'pearson', 'user_based': False}
itb_pearson = KNNBasic(sim_options=sim_options);
cv_itb_pearson = cross_validate(itb_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
x = [0,1,2]
y_mae = [cv_itb_cos['test_mae'].mean(),cv_itb_msd['test_mae'].mean(),cv_itb_pearson['test_mae'].mean()]
plt.plot(x, y_mae)
plt.title('ItemBased Collaborative Filtering for 5 fold cross validation')
plt.legend(['MAE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity measure')
plt.ylabel('Average MAE')
plt.show()
x = [0,1,2]
y_rmse = [cv_itb_cos['test_rmse'].mean(),cv_itb_msd['test_rmse'].mean(),cv_itb_pearson['test_rmse'].mean()]
plt.plot(x, y_rmse)
plt.title('ItemBased Collaborative Filtering for 5 fold cross validation')
plt.legend(['RMSE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity measure')
plt.ylabel('Average RMSE')
plt.show()
trainDaset, testDaset = train_test_split(data, test_size = 0.25, random_state = 42)
#User Based CF
usbc_nn_mae = []
usbc_nn_rmse = []
k1 = list(np.arange(1,20,1))
for i in k1:
  usbc_nn = KNNBasic(k = i, sim_options = {'user_based' : True})
  usbc_nn.fit(trainDaset)
  predictions = usbc_nn.test(testDaset)
  usbc_nn_mae.append(accuracy.mae(predictions))
  usbc_nn_rmse.append(accuracy.rmse(predictions))
plt.plot(k1,usbc_nn_mae)
plt.xlabel('Number of neighbors count')
plt.ylabel('Testset MAE')
plt.legend(['MAE'])
plt.title('UserBased collaborative filtering')
plt.show()
plt.plot(k1,usbc_nn_rmse)
plt.xlabel('Number of neighbors count')
plt.ylabel('Testset RMSE')
plt.legend(['RMSE'])
plt.title('UserBased collaborative filtering')
plt.show()
k_usbc = usbc_nn_rmse.index(min(usbc_nn_rmse))+1
print('optimum best Value of K : ', k_usbc)
print('RMSE minimum value: ', min(usbc_nn_rmse))
itbc_nn_mae = []
itbc_nn_rmse = []
for i in k1:
  itbc_nn = KNNBasic(k = i, sim_options = {'user_based' : False})
  itbc_nn.fit(trainDaset)
  predictions = itbc_nn.test(testDaset)
  itbc_nn_mae.append(accuracy.mae(predictions))
  itbc_nn_rmse.append(accuracy.rmse(predictions))
  plt.plot(k1,itbc_nn_mae)
plt.xlabel('Number of Neighbors count')
plt.ylabel('Testset MAE')
plt.legend(['MAE'])
plt.title('ItemBased collaborative filtering')
plt.show()
plt.plot(k1,itbc_nn_rmse)
plt.xlabel('Number of Neighbors count')
plt.ylabel('Testset RMSE')
plt.legend(['RMSE'])
plt.title('ItemBased collaborative filtering')
plt.show()
k_itbc = itbc_nn_rmse.index(min(itbc_nn_rmse))+1
print('Best Value of K : ', k_itbc)
print('Minimum RMSE : ', min(itbc_nn_rmse))