#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Turn interactive plotting off
plt.ioff()

#########################################################################
# 1 - Analyse descriptive des données

# read input text and put data inside a data frame
fruits = pd.read_table('data/fruit_data_with_colors.txt')
print(fruits.head())

# print nb of instances and features
print(fruits.shape)

# print feature types
print(fruits.dtypes)

# print balance between classes
print(fruits.groupby('fruit_name').size())

# plot correlation between attributes w.r.t. classification
from matplotlib import cm

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
# TODO
# y = 

# fig = plt.figure()
# cmap = cm.get_cmap('gnuplot')
# scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('fig/fruits_scatter_matrix')
# plt.close(fig)

# # print histogram for each attribute with belonging to classes
# for attr in feature_names:
#     fig = plt.figure()
#     pd.DataFrame({k: v for k, v in fruits.groupby('fruit_name')[attr]}).plot.hist(stacked=True)
#     plt.suptitle(attr)
#     plt.savefig('fig/fruits_histogram_'+attr)
#     plt.close(fig)



#########################################################################
# 2 - Prétraitement

# attr='mass'

# # discretize with equal-intervaled bins
# fig = plt.figure()
# plt.subplot(211)
# matplotlib.pyplot.xticks(fontsize=6)
# pd.cut(fruits[attr],10).value_counts(sort=False).plot.bar()
# plt.xticks(rotation=25)
# # discretize with equal-sized bins
# plt.subplot(212)
# matplotlib.pyplot.xticks(fontsize=6)
# # TODO: plot with qcut
# plt.xticks(rotation=25)
# plt.suptitle('Histogram for '+attr+' discretized with equal-intervaled and equal-sized bins')
# plt.savefig('fig/'+attr+'_histogram_discretization')
# plt.close(fig)



#########################################################################
# 3 - Cluster

# # Normalize data
# scaler = MinMaxScaler()

# # TODO
# X = 
# y = 
# X_norm = 

# # kmeans
# from sklearn.model_selection import train_test_split
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import KMeans

# # Plot clusters
# lst_kmeans = [KMeans(n_clusters=n,n_init='auto') for n in range(3,6)]
# titles = [str(x)+' clusters' for x in range(3,6)]
# fignum = 1
# for kmeans in lst_kmeans:
#     fig = plt.figure(figsize=(8, 6))
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#     kmeans.fit(X_norm)
#     labels = kmeans.labels_
#     ax.scatter(X['mass'], X['width'], X['color_score'],
#                c=labels.astype(np.float), edgecolor='k')

#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('mass')
#     ax.set_ylabel('width')
#     ax.set_zlabel('color_score')
#     ax.set_title(titles[fignum - 1])
#     plt.savefig('fig/k-means_'+str(2+fignum)+'_clusters')
#     fignum = fignum + 1
#     plt.close(fig)

# # Plot the ground truth
# fig = plt.figure(figsize=(8, 6))
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# for label in fruits['fruit_name'].unique():
#     ax.text3D(fruits.loc[fruits['fruit_name']==label].mass.mean(),
#               fruits.loc[fruits['fruit_name']==label].width.mean(),
#               fruits.loc[fruits['fruit_name']==label].color_score.mean(),
#               label,
#               horizontalalignment='center',
#               bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# ax.scatter(X['mass'], X['width'], X['color_score'], c=y, edgecolor='k')
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# ax.set_xlabel('mass')
# ax.set_ylabel('width')
# ax.set_zlabel('color_score')
# ax.set_title('Ground Truth')
# plt.savefig('fig/k-means_ground_truth')
# plt.close(fig)


# # Compute R-square, i.e. V_inter/V
# from R_square_clustering import r_square
# from purity import purity_score

# # Plot elbow graphs for KMeans using R square and purity scores
# lst_k=range(2,11)
# lst_rsq = []
# lst_purity = []
# for k in lst_k:
#     est=KMeans(n_clusters=k,n_init='auto')
#     est.fit(X_norm)
#     lst_rsq.append(r_square(X_norm, est.cluster_centers_,est.labels_,k))
#     # TODO: complete lst_purity
    
# fig = plt.figure()
# plt.plot(lst_k, lst_rsq, 'bx-')
# plt.plot(lst_k, lst_purity, 'rx-')
# plt.xlabel('k')
# plt.ylabel('RSQ/purity score')
# plt.title('The Elbow Method showing the optimal k')
# plt.savefig('fig/k-means_elbow_method')
# plt.close()
    


# # hierarchical clustering
# from scipy.cluster.hierarchy import dendrogram, linkage

# lst_labels = list(map(lambda pair: pair[0]+str(pair[1]), zip(fruits['fruit_name'].values,fruits.index)))
# linkage_matrix = linkage(X_norm, 'ward')
# fig = plt.figure()
# dendrogram(
#     linkage_matrix,
#     color_threshold=0,
#     labels=lst_labels
# )
# plt.title('Hierarchical Clustering Dendrogram (Ward)')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# plt.tight_layout()
# plt.savefig('fig/hierarchical-clustering')
# plt.close()


#########################################################################
# 4 - Classement

# from sklearn.dummy import DummyClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix


# # Create Training and Test Sets and Apply Scaling
# # by default test data represents 25%
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Normalize data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# # TODO
# X_train = 
# X_test = 


# dummycl = DummyClassifier(strategy="most_frequent")
# gmb = GaussianNB()
# dectree = tree.DecisionTreeClassifier()
# rdforest = RandomForestClassifier()
# logreg = LogisticRegression()

# lst_classif = [dummycl, gmb, dectree, rdforest, logreg]
# lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression']

# for clf,name_clf in zip(lst_classif,lst_classif_names):
#     clf.fit(X_train, y_train)
#     # TODO
#     y_pred = 

#     print('Accuracy of '+name_clf+' classifier on training set: {:.2f}'
#           .format(clf.score(X_train, y_train)))
#     print('Accuracy of '+name_clf+' classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))
#     print(confusion_matrix(y_test, y_pred))

# # print decision tree
# from sklearn import tree
# fig = plt.figure(num=None, figsize=(10, 8), dpi=300)
# tree.plot_tree(dectree,  
#                feature_names=feature_names,  
#                class_names=fruits['fruit_name'].unique(),  
#                filled=True, rounded=True)
# plt.savefig('fig/decision_tree')
# plt.close(fig)


# # Supervised learning
# # cross-validation
# from sklearn.model_selection import cross_val_score

# for clf,name_clf in zip(lst_classif,lst_classif_names):
#     # TODO : complete with function cross_val_score
#     scores = 
#     print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# # Parameters selection for Logistic regression model
# from sklearn.model_selection import GridSearchCV

# parameters={"C":[0.01,0.05,0.1,0.15,1,10],
#         "penalty":['l2',None]}
# search = GridSearchCV(LogisticRegression(), parameters, cv=5, verbose=1)
# # TODO: apply search on data
# print("Best score: %0.3f" % search.best_score_)
# print("Best parameters set:")
# best_parameters = search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))





#########################################################################
# 5 - Classement et discrétisation

# list_prefix = ['eqsized_bins_', 'eqintervaled_bins_']
# nb_bin = 10
# for prefix in list_prefix:
#     print("###### Discretization with "+prefix+" ######")
    
#     for attr in feature_names:
#         if 'sized' in prefix:
#             fruits[prefix+attr]=pd.qcut(fruits[attr],nb_bin)
#         else:
#             fruits[prefix+attr]=pd.cut(fruits[attr],nb_bin)
#         # use pd.concat to join the new columns with your original dataframe
#         fruits=pd.concat([fruits,pd.get_dummies(fruits[prefix+attr],prefix=prefix+attr)],axis=1)
#         # now drop the original column (you don't need it anymore)
#         fruits.drop(prefix+attr,axis=1, inplace=True)

#     feature_names_bins = filter(lambda x: x.startswith(prefix) and x.endswith(']'), list(fruits))
#     X_discret = fruits[feature_names_bins]
#     print(X_discret.head())

#     # TODO: compute accuracies using cross validation with the classifiers




#########################################################################
# 6 - ACP et sélection de variables

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # It is better to use standardization than normalization for PCA
# scaler = StandardScaler()
# X_standard = scaler.fit_transform(X)

# acp = PCA(svd_solver='full')
# coord = acp.fit_transform(X_standard)
# # nb of computed components
# print(acp.n_components_) 


# # explained variance scores
# exp_var_pca = acp.explained_variance_ratio_
# print(exp_var_pca)


# #
# # Cumulative sum of explained variance values; This will be used to
# # create step plot for visualizing the variance explained by each
# # principal component.
# #
# cum_sum_expl_var = np.cumsum(exp_var_pca)

# #
# # Create the visualization plot
# #
# fig = plt.figure()
# plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
# plt.step(range(0,len(cum_sum_expl_var)), cum_sum_expl_var, where='mid',label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('fig/acp_expl_var')
# plt.close(fig)


# # print eigen vectors
# print(acp.components_)

# # plot instances on the first plan (first 2 factors) or 2nd plan
# def plot_instances_acp(coord,df_labels,x_axis,y_axis):
#     fig, axes = plt.subplots(figsize=(20,20))
#     axes.set_xlim(-1.5,1.5) # limits must be manually adjusted to the data
#     axes.set_ylim(-0.8,0.8)
#     for i in range(len(df_labels.index)):
#         plt.annotate(df_labels.values[i],(coord[i,x_axis],coord[i,y_axis]))
#     plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
#     plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
#     plt.savefig('fig/acp_instances_axes_'+str(x_axis)+'_'+str(y_axis))
#     plt.close(fig)

# plot_instances_acp(coord,y,0,1)
# plot_instances_acp(coord,y,2,3)


# # compute correlations between factors and original variables
# loadings = acp.components_.T * np.sqrt(acp.explained_variance_)

# # plot correlation_circles
# def correlation_circle(components,var_names,x_axis,y_axis):
#     fig, axes = plt.subplots(figsize=(8,8))
#     minx = -1
#     maxx = 1
#     miny = -1
#     maxy = 1
#     axes.set_xlim(minx,maxx)
#     axes.set_ylim(miny,maxy)
#     # label with variable names
#     # ignore first variable (instance name)
#     for i in range(0, components.shape[1]):
#         axes.arrow(0,
#                    0,  # Start the arrow at the origin
#                    components[i, x_axis],  #0 for PC1
#                    components[i, y_axis],  #1 for PC2
#              head_width=0.01,
#              head_length=0.02)

#         plt.text(components[i, x_axis] + 0.05,
#                  components[i, y_axis] + 0.05,
#                  var_names[i])
#     # axes
#     plt.plot([minx,maxx],[0,0],color='silver',linestyle='-',linewidth=1)
#     plt.plot([0,0],[miny,maxy],color='silver',linestyle='-',linewidth=1)
#     # add a circle
#     cercle = plt.Circle((0,0),1,color='blue',fill=False)
#     axes.add_artist(cercle)
#     plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
#     plt.close(fig)

# # ignore 1st 3 columns: fruit_label, fruit_name and fruit_subtype
# correlation_circle(loadings,fruits.columns[3:],0,1)



# # Plot the covariance matrix to identify the correlation between
# # original features using a heatmap:
# covar = np.corrcoef(X_norm.T)
# fig = plt.figure(figsize=(15, 15))
# plt.matshow(covar, cmap=plt.cm.rainbow,fignum=plt.gcf().number)
# plt.colorbar(ticks = [-1, 0, 1], fraction=0.045)
# for i in range(covar.shape[0]):
#     for j in range(covar.shape[1]):
#         plt.text(i, j, "%0.2f" % covar[i,j], size=12, color='black', ha="center", va="center")
# plt.savefig('fig/correlation_original_axes')
# plt.close(fig)










# from sklearn.model_selection import cross_val_score
# from sklearn import metrics

# lst_classif = [dummycl, gmb, dectree, rdforest, logreg]
# lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression']
# print('*** Results for first 2 factors of ACP ***')
# # TODO

# print('*** Results for first 2 original variables ACP ***')
# # TODO


# # Variable selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
# selector = SelectKBest(mutual_info_classif, k=2)
# X_select = selector.fit_transform(X_norm, y)
# print('selected features: ')
# # TODO
# print('*** Results for the 2 attributes selected with mutual_info_classif ***')
# # TODO
