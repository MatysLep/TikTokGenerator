# TikTokGenerator

## Dépendances

Ce projet utilise `pytubefix` pour le téléchargement des vidéos YouTube ainsi
que `customtkinter` pour l'interface graphique. Une étape de centrage de la
vidéo s'appuie sur `mediapipe` et `opencv-python`. Une barre de progression
basée sur `tqdm` affiche l'avancement du traitement. Installez les
dépendances avec :

```bash
pip install pytubefix customtkinter mediapipe opencv-python tqdm
```

## Fonctionnalités

- Téléchargement temporaire d'une vidéo YouTube ou sélection d'un fichier local.
- Découpage fictif en clips (fonction `cut_into_clips`).
- Centrage automatique sur le locuteur avec un zoom réglable de 0 à 100 % via
  un curseur. Une étiquette affiche la valeur courante du zoom. La fonction
  `center_on_speaker` applique ce facteur et la vidéo générée est recadrée au
  format 9/16 sans bandes noires.
- Barre de progression de 3 étapes affichée au-dessus de la console de logs.
- L'interface permet de choisir entre un lien YouTube et une vidéo locale.
- Une prévisualisation du résultat est affichée avant le découpage final.


