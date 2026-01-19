# TikTokGenerator 🎬

> **Transformez n'importe quelle vidéo paysage en clips TikTok viraux, sous-titrés et recadrés par IA en quelques secondes.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)

---

## 💡 Contexte & Motivation

*Pourquoi avoir construit cet outil ?*

La création de contenu "short-form" demande une régularité extrême, mais le montage technique (sous-titrage karaoké, recadrage dynamique) est une tâche répétitive et chronophage.

J'ai développé **TikTokGenerator** pour résoudre ce problème de **scalabilité** : l'objectif était de concevoir un "monteur virtuel" capable de prendre des décisions intelligentes (suivi de visage, timing audio) sans aucune intervention humaine. Ce projet démontre comment l'IA peut transformer un workflow créatif manuel en un pipeline industriel automatisé.

---

## 🏗️ Aperçu Technique

TikTokGenerator est un pipeline de traitement vidéo automatisé conçu pour la création de contenu à grande échelle. Il orchestre des bibliothèques de **Vision par Ordinateur** et de **Traitement du Langage Naturel** via une interface **Streamlit** réactive. L'architecture suit une approche événementielle où le traitement vidéo lourd (téléchargement, analyse, montage) est géré de manière asynchrone pour ne pas bloquer l'interface utilisateur.

## 🌟 Fonctionnalités Clés

*   **🎯 Smart Crop (Recadrage Intelligent)** : Utilise **MediaPipe Face Detection** pour scanner la vidéo et déterminer dynamiquement la zone d'intérêt, transformant automatiquement le format 16:9 (YouTube) en 9:16 (TikTok) sans couper le sujet.
*   **🗣️ Sous-titres Dynamiques "Karaoké"** : Intègre **OpenAI Whisper** pour transcrire l'audio avec une précision quasi-humaine, puis génère des sous-titres stylisés (ASS) avec une animation d'apparition mot par mot pour maximiser la rétention.
*   **✂️ Segmentation Automatique** : Découpe intelligemment les longues vidéos en clips optimisés de 60 secondes, prêts à être publiés, tout en préservant la continuité audio.
*   **🌐 Sources Flexibles** : Prend en charge le téléchargement direct via **URL YouTube** (gestion des flux adaptatifs) ou l'upload de fichiers locaux (MP4, MKV, MOV).

## 🛠️ Stack Technique

| Catégorie | Technologies |
| :--- | :--- |
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **Backend / Core** | Python, Asyncio |
| **AI & Vision** | MediaPipe (Face Detection), OpenAI Whisper (ASR) |
| **Traitement Vidéo** | OpenCV, FFmpeg (via subprocess), PytubeFix |
| **Traitement Audio** | PySubs2 |

## 🚀 Installation & Usage

### Prérequis
*   Python 3.10+
*   **FFmpeg** installé et accessible dans le PATH système.

### Démarrage Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/TikTokGenerator.git
cd TikTokGenerator

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'interface sera accessible à l'adresse `http://localhost:8501`.

## 🧠 Challenge & Apprentissage

### Le Défi : Recadrage Vertical Automatisé (16:9 vers 9:16)
L'un des plus grands défis techniques a été de convertir des vidéos horizontales en format vertical sans perdre l'information visuelle essentielle (le locuteur). Un recadrage central "bête" coupait souvent les visages si le sujet n'était pas parfaitement au centre.

### La Solution : Analyse Prédictive par Computer Vision
J'ai implémenté un **système de "Smart Crop" en deux passes** :
1.  **Analyse** : Le script scanne la vidéo image par image (avec un pas d'échantillonnage) utilisant `MediaPipe` pour détecter les cadres englobants (bounding boxes) des visages.
2.  **Calcul de Marge** : Il calcule les marges de sécurité minimales à gauche et à droite sur l'ensemble de la vidéo pour définir une fenêtre de recadrage fixe optimale qui garantit que le sujet reste dans le cadre 100% du temps.
3.  **Fallback** : Si la vidéo contient plusieurs sujets écartés ou aucun visage, l'algorithme bascule intelligemment sur un fond flouté ("Gaussian Blur background") pour préserver l'esthétique sans déformer l'image origine.

---

*Projet réalisé par [Votre Nom] - 2026*