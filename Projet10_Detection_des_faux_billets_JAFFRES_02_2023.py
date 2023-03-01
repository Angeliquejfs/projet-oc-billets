#!/usr/bin/env python
# coding: utf-8

# # Partie I : Importation des données

# ## 1 ) Importation des bibliothèques

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import scipy.stats as st

import plotly.express as px

import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import svm
from sklearn import metrics
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.preprocessing import StandardScaler


# ## 2 ) Importation de fonction(s)

# In[2]:


pd.set_option("display.max_columns", None)


# ## 3 ) Importation des fichiers

# In[3]:


df_billets = pd.read_csv('billets.csv',sep = ";")
df_billets


# - length : la longueur du billet (en mm) 
# - height_left : la hauteur du billet (mesurée sur le côté gauche, en
# mm) 
# - height_right : la hauteur du billet (mesurée sur le côté droit, en mm) 
# - margin_up : la marge entre le bord supérieur du billet et l'image de
# celui-ci (en mm) 
# - margin_low : la marge entre le bord inférieur du billet et l'image de
# celui-ci (en mm) 
# - diagonal : la diagonale du billet (en mm)

# # Analyse exploratoire

# ## Préparation du fichier

# In[4]:


# Nous allons procéder à un renommage des colonnes pour une meilleure compréhension


# In[5]:


df_billets.rename(columns={"is_genuine": "Authentique","length": "Longueur","height_left":"Hauteur_gauche","height_right":"Hauteur_droite","margin_up":"Marge_sup","margin_low":"Marge_inf","diagonal":"Diagonale"}, inplace = True )
df_billets


# ## Affichage

# In[6]:


# Regardons les 5 premières lignes de ce fichier
df_billets.head()


# In[7]:


# Regardons les 5 dernières lignes de ce fichier
df_billets.tail()


# In[8]:


# Regardons un échantillon aléatoire de ce fichier
df_billets.sample(5)


# ## Structure

# In[9]:


df_billets.info()


# In[10]:


# Il s'agit bien de booléen pour les vrais/faux


# In[11]:


df_billets.shape


# In[12]:


df_billets.columns


# In[13]:


df_billets.dtypes


# In[14]:


df_billets.dtypes.value_counts()


# In[15]:


df_billets.nunique()


# In[16]:


# Nous serons sur de la classification binaire car authentique n'a bien que 2 valeurs


# In[17]:


df_billets.describe()


# In[18]:


# A priori pas de données abérantes


# ## Gestion des NAN

# L'ancien collègue en charge de ce travail s'était retrouvé avec pas mal de données manquantes qu'il a comblé par régression linéaire nous opterons peut-être pour cette méthode

# In[19]:


df_billets.isna().mean()


# In[20]:


# Nous avons donc bien des nan dans ce fichier, essentiellement sur le colonne des marges inférieures. A priori il ne s'agit pas d'un gros volume de lignes 


# In[21]:


# Vérifions quelles sont les lignes concernées par les NAN
df_billets_nan = df_billets[df_billets['Marge_inf'].isna()]
df_billets_nan 


# In[22]:


df_billets_nan['Authentique'].value_counts()


# In[23]:


# Nous avons donc dans cet extrait des billets 29 vrais et 8 faux


# In[24]:


# Vérifions la proportion de notre data de vrais et faux billets dans ce dataframe
df_billets['Authentique'].value_counts().sort_values().plot(kind = 'barh')
plt.title("Répartition du nombre de vrais et faux billets")


# In[25]:


# Nous avons donc assez de faux dans notre dataset pour potentiellement décider de les retirer, dans tous les cas nous devons les traiter car nous allons avoir des problèmes 


# Tentons de voir si nous utilisions la méthode de notre ancien collègue pour voir ce que cela donne

# #### Régression linéaire

# In[26]:


# Créons un dataframe sans nan
df_billets_sans_nan = df_billets.dropna(inplace=False)
# Calculons la matrice de corrélation
df_billets_sans_nan.corr()


# In[27]:


# On utilise la régression pour trouver les nan de la colonne Marge_inf
from sklearn.linear_model import LinearRegression


# In[28]:


y_input_nan = df_billets_sans_nan['Marge_inf']
X_input_nan = df_billets_sans_nan.drop(columns=['Marge_inf','Authentique'])


# In[29]:


y_input_nan.head()


# In[30]:


X_input_nan.head()


# In[31]:


reg = LinearRegression().fit(X_input_nan, y_input_nan)


# In[32]:


reg.score(X_input_nan, y_input_nan)


# In[ ]:





# In[33]:


X_input_nanx = sm.add_constant(X_input_nan)

model = sm.OLS(y_input_nan, X_input_nan).fit()
predictions = model.predict(X_input_nan) 

print_model = model.summary()
print(print_model)


# Les p-valeurs sont inférieures à 5 %
# À un niveau de test de 5 %, on rejette donc l'hypothèse selon laquelle le paramètre est égal à 0 : les paramètres sont donc significativement différents de 0

# In[34]:


# Analysons la régression linéaire


# In[35]:


from sklearn.metrics import r2_score


# In[36]:


y_train_predict = reg.predict(X_input_nan)


# In[37]:


# On calcule le r2 score de la régression linéaire 
r2_score(y_input_nan, y_train_predict, force_finite=False)


# In[38]:


# On effectue un test de multicolinéarité on test de VIF !!! ajout commentaires


# VIF évalue si les facteurs sont corrélés les uns aux autres (multi-colinéarité), ce qui pourrait influencer les autres facteurs et réduire la fiabilité du modèle.Si un VIF est supérieur à 10, vous avez une multi-colinéarité élevée : la variation semblera plus grande et le facteur apparaîtra plus influent qu'il ne l'est. Si VIF est plus proche de 1, alors le modèle est beaucoup plus robuste, car les facteurs ne sont pas influencés par la corrélation avec d'autres facteurs.

# In[39]:


#create DataFrame to hold VIF values
vif_df = pd.DataFrame()
vif_df['variable'] = X_input_nan.columns 
vif_df


# In[40]:


#calculate VIF for each predictor variable 
vif_df['VIF'] = [variance_inflation_factor(X_input_nan.values, i) for i in range(X_input_nan.shape[1])]
vif_df


# In[41]:


# Les résultats sont très élevés ce qui est finalement assez logique car les données concerne un produit dans l'ensemble assez standordisé il est donc logique que les variables soient corrélées entre elles 


# In[42]:


X_test = df_billets_nan.drop(columns=['Marge_inf','Authentique'])


# In[43]:


X_test.head()


# In[44]:


#vérifier les colonnes


# In[45]:


y_pred = reg.predict(X_test)
y_pred


# In[ ]:





# In[ ]:





# In[46]:


df_billets_nan['Marge_inf'] = y_pred
df_billets_nan


# In[47]:


df_concat_with_nan = df_billets_sans_nan.append(df_billets_nan, ignore_index=True)
df_concat_with_nan


# In[48]:


# Voici notre fichier final si nous décidons de remplacer les nan par une régression linéaire


# Nous choisirons d'utiliser le fichier en éliminant les lignes contenant des nan car il n'y a pas beaucoup de données en contenant de plus cela évitera de créer un biais

# - Fichier final

# In[49]:


df_billets_production = df_billets.dropna(subset=['Marge_inf'])
df_billets_production


# In[50]:


# Nous allons renommer la colonne "Authentique" avec des 0 pour False et 1 pour les True
df_billets_production['Authentique'].replace(['True','False'], [1,0], inplace=True)
df_billets_production


# In[51]:


df_billets_production.isna().mean()


# ## Analyse avec bibliothèques

# ### Dataprep

# In[52]:


create_report(df_billets_production)


# ## Analyse 

# In[53]:


# On note qu'il y a une corrélation entre la variable Longueur et Marge inf


# In[54]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = df_billets_production['Marge_inf'][:, np.newaxis]
y = df_billets_production['Longueur'][:, np.newaxis]

model = LinearRegression()

model.fit(x,y)
y_predict = model.predict(x)

rmse_linear = np.sqrt(mean_squared_error(y,y_predict))
print(rmse_linear)

plt.scatter(x, y)
plt.plot(x, y_predict, color='g')
plt.show()


# In[55]:


plt.scatter(df_billets_production['Marge_inf'], 
            df_billets_production['Marge_sup'])
plt.title('Répartition marge haute et basse ')


# In[56]:


plt.scatter(df_billets_production['Hauteur_gauche'], 
            df_billets_production['Hauteur_droite'])
plt.title('Répartition taille droite et gauche ')


# In[57]:


plt.scatter(df_billets_production['Longueur'], 
            df_billets_production['Marge_sup'])
plt.title('Répartition Longueur et marge supérieure ')


# In[58]:


df_billets_production['Diagonale'].hist()


# In[59]:


df_billets_production['Longueur'].hist()


# ## Analyse en composantes principales

# In[60]:


# Réduire la compléxité superflue d'un dataset en projetant ses données dans un espace de plus petite dimension


# In[61]:


df_billets_production_acp = df_billets_production.copy()
df_billets_production_acp_X = df_billets_production.values


# In[62]:


# Voir la forme de notre dataframe
df_billets_production_acp_X.shape


# In[63]:


std_scale = preprocessing.StandardScaler().fit(df_billets_production_acp_X)
df_billets_production_acp_X_scaled = std_scale.transform(df_billets_production_acp_X)


# In[64]:


from sklearn import decomposition

pca = decomposition.PCA(n_components=7)
pca.fit(df_billets_production_acp_X_scaled)


# In[65]:


print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[66]:


# enregistrons cette info dans une variable
scree = (pca.explained_variance_ratio_*100).round(2)
scree


# In[67]:


scree_cum = scree.cumsum().round()
scree_cum


# In[68]:


n_components = 7


# In[69]:


x_list = range(1, n_components+1)
list(x_list)


# In[70]:


plt.bar(x_list, scree)
plt.plot(x_list, scree_cum,c="red",marker='o')
plt.xlabel("rang de l'axe d'inertie")
plt.ylabel("pourcentage d'inertie")
plt.title("Eboulis des valeurs propres")
plt.show(block=False)


# In[71]:


# Le point d'inflexion semble être à 5


# - Corrélation

# In[72]:


pcs = pca.components_


# In[73]:


# Même chose qu'au dessus mais version panda
pcs = pd.DataFrame(pcs)
pcs


# In[74]:


# Enregistrer le nom de nos colonnes dans une variable également
features = df_billets_production.columns
features


# In[75]:


pcs.columns = features
pcs.index = [f"F{i}" for i in x_list]
pcs.round(2)


# In[76]:


# Affichons le data
pcs.T


# In[77]:


x, y = 0,1


# In[78]:


# Réalisation d'une fonction pour faire nos reprèsentations plus rapidement
def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # Titre
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)


# In[79]:


x_y = (0,1)
x_y


# In[80]:


correlation_graph(pca, x_y, features)


# In[81]:


correlation_graph(pca, (2,3), features)


# 

# In[82]:


# On voit bien qu'il se passe quelque chose entre la longuer et la marge inférieure et supérieure


# In[83]:


# Visualisaton graphique du fichier ci-dessus
fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(pcs.T, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")


# ## KMEANS

# In[84]:


df_billets_production_kmeans = df_billets_production.copy()


# In[85]:


df_billets_production_kmeans_X = df_billets_production_kmeans.values


# In[86]:


# on entraîne un k-means, et on enregistre la valeur de l'inertie


# In[87]:


# Une liste vide pour enregistrer les inerties :  
intertia_list = [ ]

# Notre liste de nombres de clusters : 
k_list = range(1, 10)

# Pour chaque nombre de clusters : 
for k in k_list : 
    
    # On instancie un k-means pour k clusters
    kmeans = KMeans(n_clusters=k)
    
    # On entraine
    kmeans.fit(df_billets_production_kmeans_X)
    
    # On enregistre l'inertie obtenue : 
    intertia_list.append(kmeans.inertia_)


# In[88]:


fig, ax = plt.subplots(1,1,figsize=(12,6))

ax.set_ylabel("intertia")
ax.set_xlabel("n_cluster")

ax = plt.plot(k_list, intertia_list)


# In[89]:


# 2 CLUSTER


# In[90]:


# On instancie un k-means pour 2 clusters
kmeans = KMeans(n_clusters=2)
    
# On entraine
kmeans.fit(df_billets_production_kmeans_X)


# In[91]:


kmeans.labels_


# In[92]:


# On stock les labels dans une nouvelle variable
labels = kmeans.labels_


# In[93]:


# On peut stocker nos centroids dans une nouvelle variable également : 
centroids = kmeans.cluster_centers_
centroids


# In[94]:


scaler = StandardScaler()
df_billets_production_kmeans_X_scaled = scaler.fit_transform(df_billets_production_kmeans_X)


# In[95]:


pca = PCA(n_components=2)
pca.fit(df_billets_production_kmeans_X_scaled)


# In[96]:


X_proj_kmeans = pca.transform(df_billets_production_kmeans_X_scaled)
X_proj_kmeans = pd.DataFrame(X_proj_kmeans, columns = ["PC1", "PC2"])
X_proj_kmeans[:10]


# In[97]:


fig, ax = plt.subplots(1,1, figsize=(8,7))
ax.scatter(X_proj_kmeans.iloc[:, 0], X_proj_kmeans.iloc[:, 1], c= labels, cmap="Set1")
ax.set_xlabel("F1")
ax.set_ylabel("F2")
plt.show()


# In[98]:


# On utilise bien le scaler déjà entrainé : 

centroids_proj = scaler.transform(centroids)


# In[99]:


# On définit notre figure et son axe : 
fig, ax = plt.subplots(1,1, figsize=(8,7))

# On affiche nos individus, avec une transparence de 50% (alpha=0.5) : 
ax.scatter(X_proj_kmeans.iloc[:, 0], X_proj_kmeans.iloc[:, 1], c= labels, cmap="Set1", alpha =0.5)

# On affiche nos centroides, avec une couleur noire (c="black") et une frome de carré (marker="c") : 
ax.scatter(centroids_proj[:, 0], centroids_proj[:, 1],  marker="s", c="black" )

# On spécifie les axes x et y :
ax.set_xlabel("F1")
ax.set_ylabel("F2")
plt.show()


# In[100]:


# On définit notre figure et notre axe différemment : 
fig= plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# On affiche nos points : 
ax.scatter(
    X_proj_kmeans.iloc[:, 0],
    X_proj_kmeans.iloc[:, 1],
 
    c=labels, cmap="Set1", edgecolor="k", s=40)

# On spécifie le nom des axes : 
ax.set_xlabel("F1")
ax.set_ylabel("F2")


# In[101]:


fig = px.scatter_3d(x=X_proj_kmeans.iloc[:,0], y=X_proj_kmeans.iloc[:,1],
              color=labels)
fig.show()


# ## Echantillonner les données

# In[102]:


# Determiner les paramètres


# In[103]:


y = df_billets_production['Authentique']
X = df_billets_production.drop(columns=['Authentique'])


# In[104]:


# Création d'un set de test et un set d'entraînement


# In[105]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Proportion dans train', y_train.sum() / len(y_train))
print('Proportion dans test', y_test.sum() / len(y_test))


# In[106]:


# Nous sommes sur une propotion similaire


# ### Choix des tests

# ![ml_map.png](attachment:ml_map.png)

# In[107]:


# Voici le shéma de décision pour les tests à effectuer 


# ## Régression logistique

# In[108]:


# Régression logistique
from sklearn.linear_model import LogisticRegression


# In[109]:


model_regression = LogisticRegression(random_state=42)

model_regression.fit(X_train, y_train)


# In[110]:


model_regression.score(X_train, y_train)


# In[111]:


# Nous avons une précision de 98%, c'est à dire que notre modèle prédit 98% du temps le bon résultat


# In[112]:


# Prédiction jeux de données
from sklearn.metrics import confusion_matrix

y_pred = model_regression.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[113]:


# Nous avons 319 billets qui sont vrais et ont bien été détectés comme vrais, 0 vrais billets considérés comme faux , 1 faux billets considéré comme vrai et 163 faux véritablement faux


# In[114]:


from sklearn.model_selection import learning_curve


# In[115]:


N, train_score, val_score = learning_curve(model_regression,X_train, y_train, 
                                           train_sizes = np.linspace(0.1, 1.0,10),cv=5)
print(N)
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train size')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# /!\ Attention au faux considéré vrai 

# In[116]:


from sklearn.metrics import classification_report


# - La precision et le recall sont deux métriques essentielles en classification, du fait de leur robustesse et de leur interprétabilité

# La precision est également appelée Positive Predictive Value. Elle correspond au taux de prédictions correctes parmi les prédictions positives : \begin{equation*} \frac{TP}{TP+FP} \end{equation*} Elle mesure la capacité du modèle à ne pas faire d’erreur lors d’une prédiction positive.
# Le recall est également appelé sensitivity (sensibilité), true positive rate ou encore hit rate (taux de détection). Il correspond au taux d’individus positifs détectés par le modèle : \begin{equation*} \frac{TP}{TP+FN} \end{equation*} Il mesure la capacité du modèle à détecter l’ensemble des individus positifs.

# - Le F1-score évalue la capacité d’un modèle de classification à prédire efficacement les individus positifs, en faisant un compromis entre la precision et le recall

# In[117]:


print(classification_report(y_test, y_pred, labels = labels))


# In[118]:


classification_report_regression_logistique =  classification_report(y_test, y_pred, labels = labels)


# In[ ]:





# In[ ]:





# In[119]:


from sklearn.metrics import roc_curve, auc

class_probabilities = model_regression.predict_proba(X_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr,tpr)


# In[120]:


# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")


# In[121]:


# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[122]:


print(model_regression.coef_[0])


# In[ ]:





# In[123]:


X_train.columns


# In[124]:


plt.plot(X_train.columns,model_regression.coef_[0])


# In[ ]:





# In[ ]:





# ## KNN

# In[125]:


from sklearn.neighbors import KNeighborsClassifier

# Rappel : ajouter

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Proportion dans train', y_train.sum() / len(y_train))
print('Proportion dans test', y_test.sum() / len(y_test))
# In[126]:


model_knn = KNeighborsClassifier()


# In[127]:


model_knn.fit(X_train,y_train)


# In[128]:


model_knn.predict(X_test)


# In[129]:


# Prédiction jeux de données
from sklearn.metrics import confusion_matrix

y_pred = model_knn.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[130]:


classification_report_knn =  classification_report(y_test, y_pred, labels = labels)


# In[131]:


print(classification_report_knn)


# ## KNN- Hyper paramètres

# - On test l'algo pour connaître le meilleur nombre de voisins

# In[132]:


val_score = []
for k in range (1,50):
    score = cross_val_score(KNeighborsClassifier(k),X_train,y_train, cv=5).mean()
    val_score.append(score)

plt.plot(val_score)


# In[133]:


# On remarque que le nombre optimal est 5


# In[134]:


model_knnv2 = KNeighborsClassifier(n_neighbors= 5)


# In[135]:


model_knnv2.fit(X_train,y_train)
print ('Train score:', model_knnv2.score(X_train,y_train))
print ('Test score:', model_knnv2.score(X_test,y_test))


# In[136]:


# Prédiction jeux de données
from sklearn.metrics import confusion_matrix

y_pred = model_knnv2.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[137]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score


# In[138]:


classification_report_knn_5 =  classification_report(y_test, y_pred, labels = labels)


# In[139]:


print(classification_report_knn_5)


# In[ ]:





# ## SVM

# In[140]:


#Create a svm Classifier
model_svm = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
model_svm.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model_svm.predict(X_test)


# In[141]:


# Prédiction jeux de données

y_pred = model_svm.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[142]:


classification_report_svm =  classification_report(y_test, y_pred, labels = labels)


# In[143]:


print(classification_report_svm)


# ## Forêt aléatoire

# In[144]:


#Create a Gaussian Classifier
model_foret=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
model_foret.fit(X_train,y_train)

y_pred=model_foret.predict(X_test)


# In[145]:


# Prédiction jeux de données

y_pred = model_foret.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[146]:


classification_report_foret =  classification_report(y_test, y_pred, labels = labels)


# In[147]:


print(classification_report_foret)


# ## demmyclassifier

# In[148]:


from sklearn.dummy import DummyClassifier
Estimateur = DummyClassifier(strategy = "most_frequent")
Estimateur.fit(X_train,y_train)


# In[149]:


y_prediction = Estimateur.predict(X_test)
y_prediction


# In[150]:


y_test.value_counts(normalize=True)


# In[151]:


tr_score = Estimateur.score (X_train, y_train).round(4)
ts_score = Estimateur.score(X_test, y_test).round(4)
print (f" score_train : {tr_score} score_test : {ts_score}" )


# In[152]:


def score(Estimateur) :
    tr_score = Estimateur.score (X_train, y_train).round(4)
    ts_score = Estimateur.score(X_test, y_test).round(4)
    
    print (f" score_train : {tr_score} score_test : {ts_score}" )


# In[153]:


score(Estimateur)


# In[154]:


pd.Series(y_train).value_counts(normalize=True).round(4)


# In[155]:


pd.Series(y_test).value_counts(normalize=True).round(4)


# In[156]:


mat = confusion_matrix(y_test,y_prediction)
mat


# In[157]:


mat = pd.DataFrame(mat)
mat


# In[158]:


Faux_positif_taux, Vrai_positif_taux, Thresholds = roc_curve(y_test,y_prediction)
roc_auc = auc(Faux_positif_taux, Vrai_positif_taux)
roc_auc


# In[159]:


plt.figure(figsize=(10,10))
plt.plot(Faux_positif_taux, Vrai_positif_taux, color='red',label='AUC = %0.2f' % roc_auc)


# In[160]:


# Prédiction jeux de données
from sklearn.metrics import confusion_matrix

y_pred = Estimateur.predict(X_test)
labels = [True, False]
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[161]:


classification_report_demmy =  classification_report(y_test, y_pred, labels = labels)


# In[162]:


print(classification_report_demmy)


# In[163]:


# Ajout commentaire


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Conclusion

# In[164]:


# Voici un rappel des différents classifications report


# In[165]:


print(classification_report_regression_logistique)


# In[166]:


print(classification_report_knn)


# In[167]:


print(classification_report_knn_5)


# In[168]:


print(classification_report_svm)


# In[169]:


print(classification_report_foret)


# In[170]:


print(classification_report_demmy)


# In[171]:


# Après comparaison des différents rapport on observe de meilleurs résultats avec la régression logistique


# ## Sauvegarde du modèle et interférence 

# In[172]:


filename = 'finalized_model_logistic_regression.pickle'


# In[173]:


# Permet de sauvegarder le modèle si besoin
# pickle.dump(model_regression, open(filename, 'wb'))


# In[174]:


loaded_model = pickle.load(open(filename, 'rb'))


# - Fichier avec colonne Authentique

# In[175]:


df_billets_to_predict_avec_col = pd.read_csv('billets_to_predict_aveccol.csv',sep = ";")
df_billets_to_predict_avec_col


# In[176]:


df_billets_to_predict_avec_col.rename(columns={"is_genuine": "Authentique","length": "Longueur","height_left":"Hauteur_gauche","height_right":"Hauteur_droite","margin_up":"Marge_sup","margin_low":"Marge_inf","diagonal":"Diagonale"}, inplace = True )


# In[177]:


df_billets_to_predict_avec_col.isna().mean()


# In[178]:


df_billets_to_predict_avec_col = df_billets_to_predict_avec_col.dropna(subset=['Marge_inf'])
df_billets_to_predict_avec_col


# - Fichier sans colonne Authentique

# In[179]:


df_billets_to_predict = pd.read_csv('billets_to_predict.csv',sep = ";")
df_billets_to_predict


# In[180]:


df_billets_to_predict.rename(columns={"is_genuine": "Authentique","length": "Longueur","height_left":"Hauteur_gauche","height_right":"Hauteur_droite","margin_up":"Marge_sup","margin_low":"Marge_inf","diagonal":"Diagonale"}, inplace = True )


# In[181]:


df_billets_to_predict.isna().mean()


# In[182]:


df_billets_to_predict = df_billets_to_predict.dropna(subset=['Marge_inf'])
df_billets_to_predict


# In[183]:


# Si jamais le csv comporte la vraie colonne
# X_to_predict = df_billets_to_predict.drop(columns=['Authentique'])
# Sinon X_to_predict = df_billets_to_predict


# - Test fichier sans colonne Authentique 

# In[184]:


X_to_predict = df_billets_to_predict


# In[185]:


X_to_predict


# In[186]:


y_pred = loaded_model.predict(X_to_predict)


# In[187]:


y_pred


# In[192]:


# On ne peut pas comparer avec les vraies valeurs car il n'y a pas de col auth


# In[ ]:





# - Test fichier avec colonne Authentique 

# In[191]:


y_vraie_valeur = df_billets_to_predict_avec_col['Authentique']


# In[193]:


# Prédiction jeux de données

labels = [True, False]
conf_matrix = confusion_matrix(y_vraie_valeur, y_pred, labels=labels)

print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[196]:


df_billets_to_predict_avec_col['prediction_algo'] = y_pred


# In[197]:


df_billets_to_predict_avec_col


# In[200]:


df_faux_positif = df_billets_to_predict_avec_col[df_billets_to_predict_avec_col['Authentique'] != df_billets_to_predict_avec_col['prediction_algo']]


# In[201]:


df_faux_positif


# In[203]:


df_test2 = df_billets_to_predict_avec_col[(df_billets_to_predict_avec_col['Authentique'] == False ) & (df_billets_to_predict_avec_col['prediction_algo'] == True )]


# In[204]:


df_test2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




