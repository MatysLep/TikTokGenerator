# 🎬 TikTok Clip Generator

Projet personnel de développement d'une application de génération automatique de vidéos TikTok à partir de vidéos locales ou YouTube.

## 📌 Description

**TikTok Clip Generator** est un outil automatisé développé avec **Python** et **Streamlit** permettant de créer rapidement des vidéos optimisées pour TikTok (ou autres formats verticaux).  
Il prend en entrée un **lien YouTube** ou un **fichier vidéo local** et génère automatiquement :
- Une vidéo recadrée au format 9:16 avec un **smart crop** basé sur la détection de visages
- Des **sous-titres stylisés**
- Des **clips de 61 secondes** prêts à la publication

---

## 🛠️ Fonctionnalités principales

- **Téléchargement YouTube** (vidéo + audio) avec [pytubefix](https://pypi.org/project/pytubefix/)
- **Chargement de fichiers locaux**
- **Recadrage intelligent** (smart crop) avec [MediaPipe](https://mediapipe.dev/) pour centrer les visages et remplir les marges avec un fond flouté
- **Ajout automatique de sous-titres** stylisés
- **Découpage automatique** en clips de 61 secondes avec audio synchronisé
- **Interface web interactive** via Streamlit
- **Suivi de progression en temps réel**
- **Export automatique** :
  - Vidéo finale → `~/Downloads/final`
  - Clips → `~/Downloads/clips`

---

## 🧱 Architecture du projet

- `app.py` : interface Streamlit (UI)
- `video_processor.py` : pipeline de traitement vidéo (smart crop, sous-titres, découpe)
- `utils.py` : fonctions utilitaires (génération de sous-titres, gestion des chemins, etc.)
- `requirements.txt` : dépendances Python

---

## 🔌 Technologies utilisées

- **Python**
- **Streamlit** (interface utilisateur)
- **OpenCV** (traitement d'images)
- **MediaPipe** (détection de visages)
- **FFmpeg** (manipulation audio/vidéo)
- **pytubefix** (téléchargement YouTube)

---

👤 Auteur

Matys Lepretre
Projet personnel

--- 

📄 Licence

Projet personnel – Tous droits réservés.