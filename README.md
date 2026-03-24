# 🌊 Analyse d'Imagerie Maritime : Segmentation & Classification
> **Projet de Vision par Ordinateur : Détection de navires et d'objets flottants (Paddles, Kayaks, Voiliers)**

---

## 📌 Présentation du Projet
Ce projet consiste à concevoir une chaîne complète de traitement d'images pour la surveillance maritime. L'objectif est d'isoler des objets sur l'eau, de corriger les défauts d'acquisition (luminosité, bruit) et de fournir des outils interactifs pour l'étiquetage et la visualisation des données.

### 🎯 Objectifs principaux
* **Segmentation Hybride :** Comparaison d'approches classiques (K-Means) et Deep Learning (Auto-encodeurs/CNN).
* **Correction de Verrous :** Traitement automatique des variations de luminosité et du bruit de surface.
* **Outils Métier :** Interfaces de visualisation et d'annotation manuelle avec persistance des données.

---

## 👥 Répartition du Travail

Pour ce projet, nous avons séparé le développement en deux pôles complémentaires :

### 🛠️ Adam: Algorithmes & Prétraitement (Collaborateur)
* **Gestion des Verrous :** Implémentation des filtres de luminosité (CLAHE) et normalisation du volume de données.
* **Moteur de Segmentation :** * Développement du clustering **K-Means**.
    * Architecture de l'**Auto-encodeur** pour la détection d'anomalies par reconstruction.
* **Analyse Morphologique :** Nettoyage des masques (Érosion/Dilatation) pour affiner les formes détectées.

### 🎨 Maxence: Interfaces & Intelligence Métier (Toi)
* **Développement UI (Streamlit) :** * Conception de l'interface de **Visualisation** (avec documentation Markdown intégrée).
    * Création de l'outil d'**Étiquetage Interactif**.
* **Système de Persistance :** Logique d'enregistrement des labels pour application sur de nouvelles images.
* **Analyse par Région :** Identification et classification des catégories (Voilier, Paddle, Kayak, etc.).
