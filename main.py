import customtkinter as ctk
import threading
import time
import os
import shutil
import tempfile
from pathlib import Path
from tkinter import filedialog
import subprocess
import sys
import re

import cv2
import mediapipe as mp

# Utilisation de la version patchée "pytubefix" pour éviter les erreurs 400
from pytubefix import YouTube

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO removed, 2 = +WARNING removed, 3 = only ERROR

class VideoProcessor:
    """Handle the different video processing steps."""

    def __init__(self, log_callback, progress_callback, done_callback):
        self.log = log_callback
        self.update_progress = progress_callback
        self.done_callback = done_callback
        self.output_video: str | None = None

    def download_youtube_video(self, url: str, tmp_dir : str) -> tuple[str, str, str]:
        """
        Télécharge une vidéo YouTube en 4K (si disponible) en téléchargeant séparément
        les flux vidéo et audio, puis en les fusionnant avec FFmpeg.

        :param url: URL de la vidéo YouTube.
        """
        yt = YouTube(url)
        print(f"Titre de la vidéo : {yt.title}")
        title = re.sub(r'[\\/*?:"<>|]', "_", yt.title)

        # Sélectionner le flux vidéo avec la plus haute résolution
        video_stream = yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4").order_by(
            "resolution").desc().first()
        if not video_stream:
            raise Exception("Aucun flux vidéo disponible.")

        # Sélectionner le flux audio avec le plus haut débit binaire
        audio_stream = yt.streams.filter(adaptive=True, only_audio=True, file_extension="mp4").order_by(
            "abr").desc().first()
        if not audio_stream:
            raise Exception("Aucun flux audio disponible.")

        # Définir les chemins de téléchargement
        video_path = os.path.join(tmp_dir, "video.mp4")
        audio_path = os.path.join(tmp_dir, "audio.mp4")

        # Télécharger les flux
        video_path = video_stream.download(output_path=tmp_dir)
        audio_path = audio_stream.download(output_path=tmp_dir)

        return title, video_path, audio_path

    def extract_audio(self, video_path: str, audio_output_path: str, temp_dir: str):
        # Extraire uniquement le nom de fichier
        print("Extracting audio from video.")
        filename = os.path.basename(video_path)
        print(f"Extracting audio from video : {filename}")

        # Nettoyer uniquement le nom
        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", filename)

        # Recomposer le chemin complet dans le même dossier
        new_video_path = os.path.join(temp_dir, safe_filename)

        # Renommer le fichier
        os.rename(video_path, new_video_path)

        # Lancer ffmpeg pour extraire l'audio
        command = [
            "ffmpeg",
            "-y",
            "-i", new_video_path,
            "-q:a", "0",
            "-map", "a",
            audio_output_path
        ]
        subprocess.run(command, check=True)
        return audio_output_path

    def add_audio_to_video(self, video_without_audio: str, source_audio: str, start_time: float, end_time: float) -> str:
        """
        Ajoute la portion d'audio comprise entre start_time et end_time (en secondes)
        du fichier source_audio à la vidéo spécifiée (video_without_audio) qui ne contient pas de son.
        Retourne le chemin du fichier vidéo de sortie généré avec l'audio fusionné.
        """
        # Étape 1 : construire le nom du fichier de sortie pour la vidéo avec audio
        if video_without_audio.endswith('.mp4'):
            output_video = video_without_audio.replace('.mp4', '_with_audio.mp4')
        else:
            output_video = video_without_audio + '.mp4'

        # Étape 2 : extraire le segment audio du fichier source entre start_time et end_time
        audio_segment = "temp_audio_segment.m4a"  # fichier temporaire pour le segment audio
        extract_cmd = [
            "ffmpeg", "-y",  # -y pour écraser le fichier existant si nécessaire
            "-i", source_audio,  # fichier audio source
            "-ss", str(start_time),  # heure de début en secondes
            "-to", str(end_time),  # heure de fin en secondes
            "-c", "copy",  # copie sans réencoder pour extraire exactement le segment
            audio_segment
        ]
        subprocess.run(extract_cmd, check=True)

        # Étape 3 : fusionner le segment audio extrait avec la vidéo sans audio
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", video_without_audio,  # clip vidéo sans piste audio
            "-i", audio_segment,  # segment audio extrait à insérer
            "-map", "0:v:0",  # mappe la piste vidéo de la première entrée
            "-map", "1:a:0",  # mappe la piste audio de la seconde entrée
            "-c:v", "copy",  # copie la vidéo sans réencodage pour conserver la qualité
            "-c:a", "aac",  # encode l'audio en AAC (compatible MP4)
            "-shortest",  # coupe la sortie à la durée du flux le plus court (ici l'audio, si plus court)
            output_video
        ]
        subprocess.run(merge_cmd, check=True)

        # Étape 4 : nettoyage du fichier audio temporaire
        try:
            import os
            os.remove(audio_segment)
        except OSError:
            pass

        return output_video

    def center_on_speaker(self, video_path: str, zoom_percent: int) -> str:
        self.log("Centrage de la vidéo sur le locuteur ...")

        home = Path(os.path.expanduser("~"))
        downloads_dir = None
        for name in ("Downloads", "T\xe9l\xe9chargements"):
            candidate = home / name
            if candidate.is_dir():
                downloads_dir = candidate
                break
        if downloads_dir is None:
            downloads_dir = home / "Downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)

        output_path = downloads_dir / (Path(video_path).stem + "_centered.mp4")

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_ratio = 9 / 16
        out_w, out_h = width, int(width / target_ratio)
        if out_h > height:
            out_h = height
            out_w = int(height * target_ratio)

        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        tolerance = 50
        smooth_factor = 0.2  # interpolation coefficient for smooth transition

        target_center = None
        target_size = None
        current_center = None
        current_size = None


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                cx_new = int((bbox.xmin + bbox.width / 2) * width)
                cy_new = int((bbox.ymin + bbox.height / 2) * height)
                size_new = (
                    int(bbox.width * width),
                    int(bbox.height * height),
                )

                if target_center is None:
                    target_center = (cx_new, cy_new)
                    target_size = size_new
                    current_center = target_center
                    current_size = target_size
                else:
                    tx, ty = target_center
                    if (
                        abs(cx_new - tx) > tolerance
                        or abs(cy_new - ty) > tolerance
                    ):
                        target_center = (cx_new, cy_new)
                        target_size = size_new
            elif target_center is None:
                target_center = (width // 2, height // 2)
                target_size = (out_w, out_h)
                current_center = target_center
                current_size = target_size

            if current_center is None:
                current_center = target_center
            else:
                cx, cy = current_center
                tx, ty = target_center
                current_center = (
                    int(cx + smooth_factor * (tx - cx)),
                    int(cy + smooth_factor * (ty - cy)),
                )

            if current_size is None:
                current_size = target_size
            else:
                cw, ch = current_size
                tw, th = target_size
                current_size = (
                    int(cw + smooth_factor * (tw - cw)),
                    int(ch + smooth_factor * (th - ch)),
                )


            cx, cy = current_center
            fw, fh = current_size
            crop_w = int(out_w - (zoom_percent / 100) * (out_w - fw))
            crop_h = int(out_h - (zoom_percent / 100) * (out_h - fh))
            crop_w = max(1, min(crop_w, width))
            crop_h = max(1, min(crop_h, height))

            left = max(0, min(cx - crop_w // 2, width - crop_w))
            top = max(0, min(cy - crop_h // 2, height - crop_h))

            cropped = frame[top : top + crop_h, left : left + crop_w]

            cropped = cv2.resize(cropped, (out_w, out_h))
            writer.write(cropped)

        cap.release()
        writer.release()
        face_detection.close()

        self.log(f"Vid\xe9o centr\xe9e enregistr\xe9e dans {output_path}")
        return str(output_path)

    def cut_into_clips(self, video_path: str, audio_path : str, tmp_dir : Path) -> list[str]:
        """
        Découpe la vidéo en clips de 1 minute 01 secondes maximum.

        :param video_path: Chemin de la vidéo d'entrée.
        :return: Liste des chemins vers les clips générés.
        """
        self.log("Découpage en clips de 61 secondes...")

        output_clips = []
        home = Path(os.path.expanduser("~"))
        downloads = home / "Downloads"


        # Obtenir la durée de la vidéo
        command_duration = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(command_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        total_duration = float(result.stdout.strip())
        segment_duration = 61  # secondes

        start = 0
        index = 1
        while start < total_duration:
            clip_name = f"clip_{index:02d}.mp4"
            output_path = tmp_dir / clip_name
            command_cut = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-ss", str(int(start)),
                "-t", str(segment_duration),
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(command_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Ajout de l’audio synchronisé
            clip_start = start
            clip_end = min(start + segment_duration, total_duration)
            # Créer le dossier clips juste avant de stocker le clip final avec audio
            clips_dir = downloads / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            final_clip_path = clips_dir / clip_name
            output_with_audio = self.add_audio_to_video(str(output_path), audio_path, clip_start, clip_end)
            shutil.move(output_with_audio, final_clip_path)
            output_clips.append(str(final_clip_path))
            index += 1
            start += segment_duration

        self.log(f"{len(output_clips)} clips générés dans {clips_dir}")
        return output_clips


    def process(self, source: str, zoom_percent: int, is_local: bool, title : str) -> None:
        self.log("\n--- Démarrage du traitement ---")
        tmp_dir = tempfile.mkdtemp(prefix="yt_dl_")
        title, video, audio = "", "", ""
        try:
            self.update_progress(0)

            if is_local:
                self.log(f"Vidéo locale sélectionnée : {source}")
                video = source
                audio = self.extract_audio(video, os.path.join(tmp_dir, "audio.mp4"), tmp_dir)
                print(f"audio : {audio}")
                print(f"video : {video}")
                title = title
            else:
                title,video,audio = self.download_youtube_video(source, tmp_dir)

            self.update_progress(1 / 3)

            centered = self.center_on_speaker(video, zoom_percent)
            print(f"centered : {centered} ")

            self.update_progress(2 / 3)

            self.cut_into_clips(centered, audio, Path(tmp_dir))
            self.update_progress(1)

            self.output_video = centered
            self.log("Traitement terminé")
            self.done_callback(centered)
        except Exception as exc:
            self.log(f"Erreur: {exc}")
        finally:
            if not is_local and video and os.path.exists(video):
                try:
                    shutil.rmtree(os.path.dirname(video))
                except OSError:
                    pass
            self.update_progress(0)


class App(ctk.CTk):
    """Interface graphique principale"""

    def __init__(self):
        super().__init__()
        self.title("TikTok Generator")
        self.geometry("600x400")

        self.processed_video: str | None = None

        self.source_var = ctk.StringVar(value="url")
        self.radio_url = ctk.CTkRadioButton(
            self,
            text="Lien YouTube",
            variable=self.source_var,
            value="url",
            command=self.toggle_source,
        )
        self.radio_local = ctk.CTkRadioButton(
            self,
            text="Vidéo locale",
            variable=self.source_var,
            value="local",
            command=self.toggle_source,
        )
        self.radio_url.pack(padx=10, pady=(10, 0), anchor="w")
        self.radio_local.pack(padx=10, pady=(0, 10), anchor="w")

        # Conteneur pour l'entrée ou le bouton de sélection
        self.source_frame = ctk.CTkFrame(self)
        self.source_frame.pack(fill="x")

        # Source widgets
        self.url_entry = ctk.CTkEntry(
            self.source_frame, placeholder_text="Coller le lien YouTube ici"
        )
        self.browse_button = ctk.CTkButton(
            self.source_frame, text="Parcourir", command=self.browse_file
        )
        self.video_path = None

        self.toggle_source()

        self.zoom_label = ctk.CTkLabel(self, text="Zoom sur le visage (%)")
        self.zoom_label.pack(pady=(5, 0))
        self.zoom_slider = ctk.CTkSlider(self, from_=0, to=100, command=self.update_zoom_value)
        self.zoom_slider.set(25)
        self.zoom_slider.pack(padx=10, pady=5, fill="x")
        self.zoom_value = ctk.CTkLabel(self, text="25%")
        self.zoom_value.pack()


        # Bouton de traitement
        self.process_button = ctk.CTkButton(self, text="Traitement", command=self.start_processing)
        self.process_button.pack(pady=10)

        # Boutons de prévisualisation et téléchargement
        self.preview_button = ctk.CTkButton(self, text="Prévisualisation", command=self.preview_video, state="disabled")
        self.preview_button.pack(pady=(0, 5))
        self.download_button = ctk.CTkButton(self, text="Téléchargement", command=self.save_video, state="disabled")
        self.download_button.pack()

        # Barre de progression
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=10, pady=(0, 10), fill="x")

        # Zone de logs
        self.log_text = ctk.CTkTextbox(self, state="disabled")
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.processor = VideoProcessor(self.add_log, self.set_progress, self.processing_done)

    def add_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_progress(self, value: float) -> None:
        self.progress_bar.set(value)

    def processing_done(self, path: str) -> None:
        self.processed_video = path
        self.preview_button.configure(state="normal")
        self.download_button.configure(state="normal")

    def update_zoom_value(self, value: float) -> None:
        self.zoom_value.configure(text=f"{int(value)}%")

    def toggle_source(self) -> None:
        self.url_entry.pack_forget()
        self.browse_button.pack_forget()
        if self.source_var.get() == "url":
            self.url_entry.pack(padx=10, pady=10, fill="x")
        else:
            self.browse_button.pack(padx=10, pady=10)

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")]
        )
        if path:
            self.video_path = path
            self.add_log(f"Fichier sélectionné : {path}")

    def preview_video(self) -> None:
        if not self.processed_video:
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", self.processed_video])
            elif os.name == "nt":
                os.startfile(self.processed_video)
            else:
                subprocess.Popen(["xdg-open", self.processed_video])
        except Exception as exc:
            self.add_log(f"Erreur lors de la prévisualisation : {exc}")

    def save_video(self) -> None:
        if not self.processed_video:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All", "*.*")],
        )
        if dest:
            try:
                shutil.copy(self.processed_video, dest)
                self.add_log(f"Vidéo enregistrée : {dest}")
            except Exception as exc:
                self.add_log(f"Erreur lors de la copie : {exc}")

    def start_processing(self) -> None:
        source_type = self.source_var.get()
        is_local = source_type == "local"

        if is_local:
            if not self.video_path:
                self.add_log("Veuillez sélectionner une vidéo locale.")
                return
            source = self.video_path
        else:
            source = self.url_entry.get().strip()
            if not source:
                self.add_log("Veuillez entrer un lien valide.")
                return

        zoom = int(self.zoom_slider.get())
        self.preview_button.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.processed_video = None
        title = "video" if is_local else "youtube_video"  # ou génère un vrai titre
        threading.Thread(
            target=self.processor.process,
            args=(source, zoom, is_local,title),
            daemon=True,
        ).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
