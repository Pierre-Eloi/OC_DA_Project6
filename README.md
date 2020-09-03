# Parcours Data Analyst : Projet 6

**objectif : réaliser une classification binaire pour identifier des faux billets**

L'analyse et les résultats sont présentés sous forme d'un notebook jupyter.  
Les librairies python nécessaires pour pouvoir lancer le notebook sont regroupées dans le fichier txt requirements.

Toutes les fonctions créées afin de mener à bien le projet ont été regroupées dans le fichier functions.

Étant donné que le jeu de données que j'avais à ma disposition était étiqueté, j'ai utilisé une **regression logistique régularisée** (Ridge) pour effectuer la classification binaire. Le grand avantage de la régression logistique étant de donner les probabilités, il faut cependant s'assurer que les données sont bien linéaires.

Lorsque l'on souhaite construire un système de détection d'anomalies, généralement très peu des données sont étiquetés. On utilise alors des algorithmes de *clustering* afin d'obtenir des clusters, puis on regarde où sont situées les observations avec étiquette pour caractériser les clusters. C'est pourquoi j'ai aussi testé l'algorithme **k-means** sur les données. Les données étant linéaires, j'ai effectué une **Analyse en Composantes Principales (ACP)** pour visualiser les résultats de la régression logistique et du k-means.
