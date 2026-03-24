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

## 🏷️ Annotation Manuelle et Transfert

1. Ouvrir l'onglet **Annotation manuelle** depuis l'accueil ou la visualisation.
2. Charger des **régions automatiques** (approche région) ou des labels déjà sauvegardés.
3. Corriger les classes (`bateau_moteur`, `voilier`, `paddle`, `kayak`, `gonflable`, `inconnu`).
4. Sauvegarder les labels.
5. Utiliser **Appliquer et adapter échelle** pour transférer les annotations sur une autre image.

## 🌍 Apprentissage Terre / Mer / Ciel

1. Ouvrir `/annotation`.
2. Cliquer **Basculer mode region/point** pour passer en mode `point`.
3. Choisir la classe scene (`terre`, `mer`, `ciel`) puis cliquer sur l'image pour poser des sélections.
4. Cliquer **Envoyer selections et apprendre**.
5. Cliquer **Predire terre/mer/ciel** pour générer la carte automatique.

Le modèle apprend en continu avec toutes les sélections sauvegardées.

## 📊 Interprétation des Résultats

- **Masque de segmentation :** Les pixels de même couleur appartiennent au même cluster
- **Luminance :** Plus claire = cluster de faible densité
- **Formes détectées :** Les contours nets indiquent une bonne séparation
- **Bruits :** Petits pixels isolés = résidu ou détails fins