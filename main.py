import customtkinter as ctk
import threading
import time
import os
import shutil
import tempfile

# Utilisation de la version patchée "pytubefix" pour éviter les erreurs 400
from pytubefix import YouTube



class VideoProcessor:
    """Stub class handling video processing steps."""

    def __init__(self, log_callback):
        self.log = log_callback

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

    def center_on_speaker(self, video_path: str) -> str:
        self.log("Centrage de la vidéo sur le locuteur ...")
        time.sleep(1)
        return "centered_" + video_path

    def cut_into_clips(self, video_path: str) -> list[str]:
        self.log("Découpage de la vidéo en clips ...")
        time.sleep(1)
        return ["clip_01.mp4"]

    def process(self, url: str) -> None:
        self.log("\n--- Démarrage du traitement ---")
        video = None
        try:
            video = self.download_video(url)
            centered = self.center_on_speaker(video)
            self.cut_into_clips(centered)
            self.log("Traitement terminé")
        except Exception as exc:
            self.log(f"Erreur: {exc}")
        finally:
            if video and os.path.exists(video):
                try:
                    shutil.rmtree(os.path.dirname(video))
                except OSError:
                    pass


class App(ctk.CTk):
    """Interface graphique principale"""

    def __init__(self):
        super().__init__()
        self.title("TikTok Generator")
        self.geometry("600x400")

        # Champ URL
        self.url_entry = ctk.CTkEntry(self, placeholder_text='Coller le lien YouTube ici')
        self.url_entry.pack(padx=10, pady=10, fill="x")

        # Bouton de traitement
        self.process_button = ctk.CTkButton(self, text="Télécharger et Traiter", command=self.start_processing)
        self.process_button.pack(pady=10)

        # Zone de logs
        self.log_text = ctk.CTkTextbox(self, state="disabled")
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.processor = VideoProcessor(self.add_log)

    def add_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_processing(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self.add_log("Veuillez entrer un lien valide.")
            return

        threading.Thread(target=self.processor.process, args=(url,), daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
