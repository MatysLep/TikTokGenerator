from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from tkinter import filedialog

import customtkinter as ctk

from video_processor import VideoProcessor


class App(ctk.CTk):
    """Interface graphique principale."""

    def __init__(self):
        super().__init__()
        self.title("TikTok Generator")
        self.geometry("600x400")

        self.processed_video: str | None = None

        self.url_entry = ctk.CTkEntry(self, placeholder_text="Coller le lien YouTube ici")
        self.url_entry.pack(padx=10, pady=10, fill="x")

        self.zoom_label = ctk.CTkLabel(self, text="Zoom sur le visage (%)")
        self.zoom_label.pack(pady=(5, 0))
        self.zoom_slider = ctk.CTkSlider(self, from_=0, to=100, command=self.update_zoom_value)
        self.zoom_slider.set(25)
        self.zoom_slider.pack(padx=10, pady=5, fill="x")
        self.zoom_value = ctk.CTkLabel(self, text="25%")
        self.zoom_value.pack()

        self.process_button = ctk.CTkButton(self, text="Traitement", command=self.start_processing)
        self.process_button.pack(pady=10)

        self.preview_button = ctk.CTkButton(self, text="Prévisualisation", command=self.preview_video, state="disabled")
        self.preview_button.pack(pady=(0, 5))
        self.download_button = ctk.CTkButton(self, text="Téléchargement", command=self.save_video, state="disabled")
        self.download_button.pack()

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=10, pady=(0, 10), fill="x")

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

    def preview_video(self) -> None:
        if not self.processed_video:
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", self.processed_video])
            elif os.name == "nt":
                os.startfile(self.processed_video)  # type: ignore[attr-defined]
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
        source = self.url_entry.get().strip()
        if not source:
            self.add_log("Veuillez entrer un lien valide.")
            return

        zoom = int(self.zoom_slider.get())
        self.preview_button.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.processed_video = None

        threading.Thread(
            target=self.run_async_process,
            args=(source, zoom),
            daemon=True,
        ).start()

    def run_async_process(self, source: str, zoom: int) -> None:
        import asyncio
        asyncio.run(self.processor.process(source, zoom))
