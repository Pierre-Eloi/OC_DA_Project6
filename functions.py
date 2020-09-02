
# coding: utf-8

'''Contient toutes les fonctions de visualisation qui seront utilisées dans le projet 6.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from IPython.display import display


def display_qual_var(data, feat_name):
    '''Permet de visualiser les variables qualitatives sous forme de tableau.
    Prend en paramètres :
    - data : le jeu de données
    - feat_name : Le nom de la variqble qulitative à visualiser
    '''
    eff = data[feat_name].value_counts()
    mods = eff.index
    df = pd.DataFrame(mods, columns=[feat_name])
    df['n'] = eff.values 
    df['f (%)'] = round(df['n'] / df['n'].sum() * 100, 1)
    display(df)

def display_hist(data, features, bins):
    '''Trace l'histogramme de chaque variable.
    Prend en paramètres :
    - data : le jeu de données
    - features : liste des variables quantitatives
    - bins : nombre de bins par histogramme
    '''
    # initialisation de la figure
    feat_nb = len(features)
    n_row = 0
    if feat_nb % 3 == 0:
        n_row = feat_nb // 3
    else:
        n_row = feat_nb // 3 + 1 
    fig = plt.figure(figsize=(18, n_row*6))
    # On crée les sousplots
    for i in range(feat_nb):
        ax = fig.add_subplot(n_row, 3, i+1)
        h = ax.hist(data[features[i]], bins=bins, edgecolor='none')
        ax.set_title(features[i])
        ax.grid(linestyle='dashed')
    plt.suptitle("Histogramme des variables quantitatives", fontsize=20)
    plt.savefig("Hist.png")
    plt.show()


def sort_by_categ(data, var_qual):
    '''Permet de classer des variables quantitatives en fonction d'une variable qualitative
    pour pouvoir ensuite tracer des boxplots.
    Les données classées sont regroupées dans un dictionnaire avec pour clés les noms des variables quantitatives.
    La fonction prend en paramètre :
    - data : le dataframe contenant les variables quantitatives et la variable qualitative
    - var_qual : le nom de la variable qualitative
    Elle renvoie le dictionnaire contenant les variables classées et le nom des catégorie
    '''
    # Une copie du jeu de données est créée en mettant en premier la variable qualitative
    var = data.columns.tolist()
    i = var.index(var_qual)
    var = var[i:i+1] + var[:i] + var[i+1:]
    df = data[var].copy()
    # Les données quantitatives sont classées en fonction de la variable qualitative
    features = var[1:]
    feat_nb = len(features)
    list_categ = df[var_qual].unique().tolist()
    dic_categ = {}
    for var in features:
        categ=[]
        for c in list_categ :
            values = df[df[var_qual]==c][var]
            cat = {'valeurs':values.values,'quantile25':values.quantile(0.25), 'std':values.std(ddof=1), 'taille':values.size}
            categ.append(cat)
        dic_categ[var] = categ
    # Le dictionnaire contenant les données classées est retournée
    return dic_categ, list_categ


def display_boxplot(dic_data, list_categ, var_qual, n):
    '''Permet de visualiser des variables quantitatives en fonction d'une variable qualitative à l'aide de boxplot,
    doit être utilisée après la fonction sort_by_categ
    Prend en paramètre :
    - dic_data : le dictionnaire contenant les données classées créées par la fonction sort_by_categ 
    - list_categ : la liste contenant le nom des catégories (2ème élément renvoyé par la fonction sort_by_categ)
    - var_qual : le nom de la variable qualitative
    - n : le nombre de plots par ligne
    '''
    # initialisation de la figure
    features = [key for key in dic_data]
    feat_nb = len(features)
    n_row = 0
    if feat_nb % n == 0:
        n_row = feat_nb // n
    else:
        n_row = feat_nb // n + 1
    fig = plt.figure(figsize=(6*n, n_row*6))
    # On paramètre l'affichage des boxplots
    medianprops = {'color':'black'}
    meanprops = {'marker':'o', 'markeredgecolor':'black', 'markeredgewidth':1, 'markerfacecolor':'peachpuff'}
    flierprops = {'marker':'+', 'markeredgecolor':'black', 'markeredgewidth':1}
    # On crée les sousplots
    for i, var in enumerate(features):
        ax = fig.add_subplot(n_row, n, i+1)
        ax.boxplot([c['valeurs'] for c in dic_data[var]], labels=list_categ, vert=False, showfliers=True, showmeans=True,
            patch_artist=True, medianprops=medianprops, meanprops=meanprops, flierprops=flierprops)
        ax.set_xlabel(var)
        ax.set_ylabel(var_qual)
        ax.set_title("{} vs. {}".format(var, var_qual), fontsize=20)
        ax.grid(linestyle='dashed')
        # On affiche l'effectif et l'écart type de chaque catégorie
        for k, c in enumerate(dic_data[var]):
            ax.text(c['quantile25'], k + 1.2,'(n={}, std={})'.format(c['taille'], round(c['std'],2)),
             verticalalignment='center', fontsize=14)       
    plt.tight_layout()
    plt.savefig("boxplot.png")
    plt.show()


def anova_sum_squares(data, var_qual, features):
    ''' Renvoie un tableau comportant le degré de liberté et la somme des carrés
    de chaque variation :
    1. variation totale
    2. variation interclasse
    3. variation intraclasse
    Prend en paramètres :
    - data : le jeu de données
    - var_qual : le nom de la variable qualitative
    - features : la liste des variables quantitatives
    '''
    x = data[var_qual]
    n = data.shape[0]
    p = x.unique().size
    index = ["variation totale", "variation interclasse", "variation intraclasse"]
    # Initialisation de la table avec les degrés de liberté pour chaque variation
    df = pd.DataFrame([n-1, p-1, n-p], index=index, columns=["DL"])
    for f in features:
        y = data[f]
        mu = y.mean()
        cat = []
        for c in x.unique():
            yi = y[x==c]
            cat.append({'ni': yi.size, 'mu_i': yi.mean()})
        SCT = sum([(yj - mu)**2 for yj in y])
        SCE = sum([c['ni']*(c['mu_i'] - mu)**2 for c in cat])
        SCR = SCT - SCE
        SS = [SCT, SCE, SCR]
        df[f] = SS
    display(df)
    return df


def anova_Ftest(df):
    ''' Permet d'effectuer un test d'analyse de variance en renvoyant une comportant :
    1. eta_2
    2. Fcalc
    3. la Pvaleur
    Prend en paramètres :
    - df : la table avec les degrés de liberté et sommes des carrés (renvoyée par anova_sum_squares)
    '''
    features = df.columns[1:].tolist()
    index = ['eta_2', 'F', 'p']
    # Initialisation de la table
    test = pd.DataFrame(np.zeros((3, len(features))), index=index, columns=features)                       
    for var in features:
        eta_2 = df[var][1] / df[var][0] 
        F = (df[var][1]/df.iloc[1, 0]) / (df[var][2]/df.iloc[2, 0])
        # F suit une loi de Fisher on utilise donc le module f de stats
        # La méthode sf est équivalente à 1 - fonction de répartition donc à p
        # (probabilité d'avoir une valeur supérieur ou égale à F)
        p = stats.f.sf(F, df.iloc[1, 0], df.iloc[2, 0])
        test[var] = [eta_2, F, p]
    display(test)


def display_scree_plot(pca):
    '''Permet la visualisation du diagramme des Éboulis.'''
    scree = pca.explained_variance_ratio_*100
    n = scree.size
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.bar(np.linspace(1, n, n), scree)
    ax.plot(np.linspace(1, n, n), scree.cumsum(),c='k', marker='o')
    ax.set_xlabel("rang de l'axe d'inertie")
    ax.set_ylabel("pourcentage d'inertie")
    ax.set_title("Eboulis des valeurs propres", fontsize=20)
    # On affiche l'inertie expliquée par chaque axe
    for i in range(n):
        ax.text(i+1, scree[i]+2,"{} %".format(round(scree[i],1)),
                    horizontalalignment='center', fontsize=14)      
    plt.savefig("Eboulis.png")
    plt.show(block=False)

def display_circles(pca, features, n):
    '''Permet la visualisation des cercles de corrélations.
    Prend en paramètres :
    - pca : une ACP effectuée sur un set de données
    - features : liste des variables initiales
    - n : Le nombre de cerles par ligne
    '''
    pcs = pca.components_
    # initialisation de la figure
    n_comp = pcs.shape[0]
    n_circles = n_comp // 2
    n_comp = n_circles * 2 # On affiche que les plans factoriels complets
    n_row = 0
    if n_circles % n == 0:
        n_row = n_circles // n
    else:
        n_row = n_circles // n + 1
    fig = plt.figure(figsize=(18, n_row*18//n*0.9))
    # Création des sous plots
    for i in range(0, n_comp, 2):
        ax = fig.add_subplot(n_row, n, i/2+1)
        ax.set_title("Cercle des corrélations (F{} et F{})".format(i+1, i+2), fontsize=20)
        # affichage des flèches et du nom des variables
        for k, (x, y ) in enumerate(zip(pcs[i, :], pcs[i+1, :])):
            ax.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[i,:], pcs[i+1,:], 
                   angles='xy', scale_units='xy', scale=1, color='firebrick', alpha=0.5)
            ax.text(x, y, features[k], fontsize='14', alpha=0.8)
        # affichage du cercle
        circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
        ax.add_artist(circle)
        # affichage des lignes horizontales et verticales
        ax.plot([-1, 1], [0, 0], color='grey', ls='--')
        ax.plot([0, 0], [-1, 1], color='grey', ls='--')
        # définition des limites du graphique
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # nom des axes, avec le pourcentage d'inertie expliqué
        ax.set_xlabel("F{} ({}%)".format(i+1, round(100*pca.explained_variance_ratio_[i],1)))
        ax.set_ylabel("F{} ({}%)".format(i+2, round(100*pca.explained_variance_ratio_[i+1],1)))
    plt.savefig("cercle_corr.png")
    plt.show()


def display_factorial_planes(data, X_scaled, pca, n, var_qual = None, save = None):
    ''' Permet la visualisation des individus dans les différents plans factoriels en fonction d'une variable qualitative.
    Prend en paramètres :
    - data : le jeu de données
    - X_scaled : les données centrées
    - pca : une ACP effectuée sur X_scaled
    - n : nombre de graphiques par lignes
    - var_qual : le nom de la variable qualitative
    '''
    # On projette les données sur les composantes principales de l'ACP  
    X_projected = pca.transform(X_scaled)
    pcs = pca.components_
    n_comp = pcs.shape[0]
    n_planes = n_comp // 2
    n_comp = n_planes * 2 # On affiche que les plans factoriels complets
    # initialisation de la figure
    n_row = 0
    if n_planes % n == 0:
        n_row = n_planes // n
    else:
        n_row = n_planes // n + 1
    if var_qual == None:
        ax_width = 18//n
    else:
        ax_width = 18//(n+0.5) # pour obtenir des plots carrés même avec la colorbar
    fig = plt.figure(figsize=(18, n_row*ax_width))
    # Création des sous plots
    for i in range(0, n_comp, 2):
        ax = fig.add_subplot(n_row, n, i/2+1)
        ax.set_title("Projection des individus (sur F{} et F{})".format(i+1, i+2), fontsize=20)
        # affichage des points
        if var_qual == None:
            ax.scatter(X_projected[:, i], X_projected[:, i+1], alpha=0.7)
        else:
            label = data[var_qual].values
            scatter = ax.scatter(X_projected[:, i], X_projected[:, i+1], c=label, cmap='Spectral')
            plt.colorbar(scatter, ticks=[0, 1])
        # affichage des lignes horizontales et verticales
        ax.plot([-100, 100], [0, 0], color='grey', ls='--')
        ax.plot([0, 0], [-100, 100], color='grey', ls='--')
        # détermination des limites du graphique
        boundary = np.max(np.abs(X_projected[:, [i,i+1]])) * 1.1
        ax.set_xlim([-boundary,boundary])
        ax.set_ylim([-boundary,boundary])
        # nom des axes, avec le pourcentage d'inertie expliqué
        ax.set_xlabel("F{} ({}%)".format(i+1, round(100*pca.explained_variance_ratio_[i],1)))
        ax.set_ylabel("F{} ({}%)".format(i+2, round(100*pca.explained_variance_ratio_[i+1],1)))
    if save == True:
        plt.savefig("factorial_planes.png")
    plt.show()


def display_clusters(X_scaled, pca, cls, clusters, labels):
    ''' Permet de visualiser les groupes dans le premier plan factoriel.
    Prend en paramètres :
    - X_scaled : les données centrées
    - pca : une ACP effectuée sur X_scaled
    - cls : un k-means effectué sur X_scaled
    - clusters : groupe assigné à chaque individu
    - labels : liste indiquant si les individus sont bien classés
    '''
    X_projected = pca.transform(X_scaled)
    X_centro_trans = pca.transform(cls.cluster_centers_)
    # initialisation de la figure
    fig = plt.figure(figsize=(18, 7))
    # affichage premier graphique
    ax1 = fig.add_subplot(121)
    ax1.set_title("Visualisation des groupes dans le 1er plan factoriel de l'ACP", fontsize=20)
    # affichage des points
    scatter1 = ax1.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters, cmap='Spectral')
    plt.colorbar(scatter1, ticks=[0, 1])
    # affichage des centroïdes
    scatter2 = ax1.scatter(X_centro_trans[:, 0], X_centro_trans[:, 1], c='k', marker='s', s=100)
    ax1.legend((scatter1, scatter2),("Attributs", "Centroïdes"), loc='upper left')
    # affichage des lignes horizontales et verticales
    ax1.plot([-100, 100], [0, 0], color='grey', ls='--')
    ax1.plot([0, 0], [-100, 100], color='grey', ls='--')
    # détermination des limites du graphique
    boundary = np.max(np.abs(X_projected[:, [0,1]])) * 1.1
    ax1.set_xlim([-boundary,boundary])
    ax1.set_ylim([-boundary,boundary])
    # nom des axes, avec le pourcentage d'inertie expliqué
    ax1.set_xlabel("F1 ({}%)".format(round(100*pca.explained_variance_ratio_[0],1)))
    ax1.set_ylabel("F2 ({}%)".format(round(100*pca.explained_variance_ratio_[1],1)))
    # affichage deuxième graphique
    ax2 = fig.add_subplot(122)
    ax2.set_title("Visualisation des individus mal classés", fontsize=20)
    # affichage des points
    scatter3 = ax2.scatter(X_projected[:, 0], X_projected[:, 1], c=labels, cmap='Spectral')
    plt.colorbar(scatter3, ticks=[0, 1])
    # affichage des lignes horizontales et verticales
    ax2.plot([-100, 100], [0, 0], color='grey', ls='--')
    ax2.plot([0, 0], [-100, 100], color='grey', ls='--')
    # détermination des limites du graphique
    boundary = np.max(np.abs(X_projected[:, [0,1]])) * 1.1
    ax2.set_xlim([-boundary,boundary])
    ax2.set_ylim([-boundary,boundary])
    # nom des axes, avec le pourcentage d'inertie expliqué
    ax2.set_xlabel("F1 ({}%)".format(round(100*pca.explained_variance_ratio_[0],1)))
    ax2.set_ylabel("F2 ({}%)".format(round(100*pca.explained_variance_ratio_[1],1)))
    
    plt.savefig("clusters.png")
    plt.show()

def my_roc_curve(y_test, y_prob):
    ''' Permet de mesurer la performance d'un classifieur binaire en visualisant la courbe ROC.
    Prend en paramètres :
    y_test : vecteur contenant les étiquettes
    y_prob : vecteur contenant les résultats du modèle
    '''
    # Calcul de la courbe ROC
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_prob)
    # Calcul de l'aire sous la courbe pour connaître l'efficacité du modèle
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    # Initialissation de la figure
    plt.figure(figsize=(7, 7))
    plt.title('Courbe ROC')
    
    # Courbe du modèle
    plt.plot(false_positive_rate,true_positive_rate,label = 'AUC = {:0.2%}'.format(roc_auc))
    # Classificateur aléatoire
    plt.plot([0, 1], [0, 1],linestyle='--', color='grey')

    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def test_billet(data, std_scale, model):
    ''' Permet de déterminer si un billet est vrai ou non.
    Prend en paramètres :
    data_scaled : le dataset contenant les caractéristiques des billets
    std_scale : l'algorithme permettant de standardiser les données
    model : le modèle sélectionné
    '''
    X = std_scale.transform(data.iloc[:, :-1].values)
    y_prob = model.predict_proba(X)[:,1]
    y_pred = model.predict(X)
    for i, id in enumerate(data['id']):
        if y_pred[i] == 1:
            print("Le billet {} est vrai avec une probabilité de {:0.2%}".format(id, y_prob[i]))
        else:
            print("Le billet {} est faux avec une probabilité de {:0.2%}".format(id, 1-y_prob[i]))   

