import customtkinter as ctk
import threading
import time
import os
import shutil
import tempfile
from pathlib import Path
from tkinter import filedialog

import cv2
import mediapipe as mp

# Utilisation de la version patchée "pytubefix" pour éviter les erreurs 400
from pytubefix import YouTube



class VideoProcessor:
    """Stub class handling video processing steps."""

    def __init__(self, log_callback, progress_callback, preview_callback):
        self.log = log_callback
        self.update_progress = progress_callback
        self.preview_ready = preview_callback

    def download_video(self, url: str) -> str:
        """Download the YouTube video to a temporary location."""
        self.log(f"Téléchargement de la vidéo depuis {url} ...")

        tmp_dir = tempfile.mkdtemp(prefix="yt_dl_")
        try:
            yt = YouTube(url)
            stream = (
                yt.streams.filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )

            if stream is None:
                raise RuntimeError("Aucune vidéo MP4 disponible")

            self.log(f"Téléchargement de {stream.default_filename} ...")
            video_path = stream.download(output_path=tmp_dir)
            return video_path
        except Exception:
            # Clean up temporary directory if download fails
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

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
        face_center = None
        face_size = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * width)
                cy = int((bbox.ymin + bbox.height / 2) * height)
                face_center = (cx, cy)
                face_size = (
                    int(bbox.width * width),
                    int(bbox.height * height),
                )

            if face_center is None:
                face_center = (width // 2, height // 2)
                face_size = (out_w, out_h)

            cx, cy = face_center
            fw, fh = face_size
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

    def preview_video(self, video_path: str) -> None:
        """Display a simple video preview."""
        self.log("Ouverture de la prévisualisation. Appuyez sur 'q' pour quitter.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Impossible d'ouvrir {video_path}")
            return

        window = "Prévisualisation"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(window, frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyWindow(window)

    def cut_into_clips(self, video_path: str) -> list[str]:
        self.log("Découpage de la vidéo en clips ...")
        time.sleep(1)
        return ["clip_01.mp4"]

    def process(self, source: str, zoom_percent: int, is_local: bool) -> None:
        self.log("\n--- Démarrage du traitement ---")
        video = None
        try:
            self.update_progress(0)

            if is_local:
                self.log(f"Vidéo locale sélectionnée : {source}")
                video = source
            else:
                video = self.download_video(source)

            self.update_progress(1 / 3)

            centered = self.center_on_speaker(video, zoom_percent)
            self.preview_ready(centered)
            self.update_progress(2 / 3)

            self.cut_into_clips(centered)
            self.update_progress(1)

            self.log("Traitement terminé")
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
        self.preview_path = None

        self.toggle_source()

        self.zoom_label = ctk.CTkLabel(self, text="Zoom sur le visage (%)")
        self.zoom_label.pack(pady=(5, 0))
        self.zoom_slider = ctk.CTkSlider(self, from_=0, to=100, command=self.update_zoom_value)
        self.zoom_slider.set(25)
        self.zoom_slider.pack(padx=10, pady=5, fill="x")
        self.zoom_value = ctk.CTkLabel(self, text="25%")
        self.zoom_value.pack()


        # Bouton de traitement
        self.process_button = ctk.CTkButton(self, text="Télécharger et Traiter", command=self.start_processing)
        self.process_button.pack(pady=10)

        # Bouton de prévisualisation
        self.preview_button = ctk.CTkButton(self, text="Prévisualiser", command=self.show_preview, state="disabled")
        self.preview_button.pack(pady=(0, 10))

        # Barre de progression
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=10, pady=(0, 10), fill="x")

        # Zone de logs
        self.log_text = ctk.CTkTextbox(self, state="disabled")
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.processor = VideoProcessor(self.add_log, self.set_progress, self.on_preview_ready)

    def add_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_progress(self, value: float) -> None:
        self.progress_bar.set(value)

    def update_zoom_value(self, value: float) -> None:
        self.zoom_value.configure(text=f"{int(value)}%")

    def on_preview_ready(self, path: str) -> None:
        def enable():
            self.preview_path = path
            self.preview_button.configure(state="normal")
            self.add_log(f"Prévisualisation disponible : {path}")

        self.after(0, enable)

    def show_preview(self) -> None:
        if self.preview_path:
            threading.Thread(target=self.processor.preview_video, args=(self.preview_path,), daemon=True).start()

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

    def start_processing(self) -> None:
        self.preview_button.configure(state="disabled")
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
        threading.Thread(target=self.processor.process, args=(source, zoom, is_local), daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
