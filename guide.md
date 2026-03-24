# 📖 Guide d'Utilisation

## 🎯 Démarrage Rapide

1. Accédez à la **Galerie** depuis l'accueil
2. Sélectionnez une image maritime
3. Choisissez un algorithme de segmentation
4. Réglez le paramètre **k** (nombre de clusters)
5. Cliquez sur **Lancer la Segmentation**

## 🤖 Algorithmes Disponibles

### K-Means (Classique)
**Description :** Clustering par partitionnement itératif. Rapide et efficace pour les images bien contrastées.

**Avantages :**
- Temps de calcul rapide
- Résultats stables

**Inconvénients :**
- Sensible aux bruits
- Nécessite de spécifier k

**Paramètre k :** Nombre de classes (2-20 recommandé)

### DBSCAN (Hybride)
**Description :** Clustering basé sur la densité. Détecte automatiquement le nombre de clusters.

**Avantages :**
- Détecte les clusters de formes arbitraires
- Robuste au bruit

**Inconvénients :**
- Plus lent que K-Means

**Paramètre k :** Rayon de voisinage (eps)

### Auto-encodeur (CNN)
**Description :** Réseau de neurones convolutif pour la segmentation. Approche deep learning.

**Avantages :**
- Très précis
- Apprentissage des features complexes

**Inconvénients :**
- Temps de calcul plus long
- Nécessite d'énormes données d'entraînement

**Paramètre k :** Nombre de couches latentes

## ⚙️ Paramètres Recommandés

| Condition | Algorithme | k Recommandé |
|-----------|------------|--------------|
| Images claires, bien contrastées | K-Means | 5-8 |
| Images avec bruits, objets multiples | DBSCAN | 3-6 |
| Images complexes, haute précision requise | Auto-encodeur | 8-15 |

## 📊 Interprétation des Résultats

- **Masque de segmentation :** Les pixels de même couleur appartiennent au même cluster
- **Luminance :** Plus claire = cluster de faible densité
- **Formes détectées :** Les contours nets indiquent une bonne séparation
- **Bruits :** Petits pixels isolés = résidu ou détails fins