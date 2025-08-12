from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from pytubefix import YouTube

import utils
from utils import get_downloads_dir, safe_rmtree

# Reduce TensorFlow logging from MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class VideoProcessor:
    def montage_smart_crop(
            self,
            output_size: tuple[int, int] = (1080, 1920),
            min_confidence: float = 0.60,
            margin_px: int = 50,
            detection_step: int = 5,
            progress_log_step: float = 0.10,
    ) -> None:
        """
        Nouveau montage: *crop intelligent constant* sur toute la vidéo.
        - Détecte les visages sur des frames échantillonnées (detection_step).
        - Calcule les marges minimales (garde-fous) gauche/droite/haut/bas sur l'ensemble.
        - Applique un *crop horizontal symétrique* maximum autorisé (avec marge de 50 px) sans couper les visages.
        - Conserve les proportions et **remplit toute la largeur** du canvas 9:16. Si la hauteur ne suffit pas, fond flou derrière.
        """
        self.log("Démarrage du montage smart-crop…")
        in_path = self.video_path
        assert in_path, "Chemin vidéo introuvable"

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ow, oh = output_size

        base = Path(in_path).stem
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_video = os.path.join(self.tmp_dir, f"{base}_smartcrop.mp4")
        out = cv2.VideoWriter(self.output_video, fourcc, fps, output_size)

        # --- Détecteur visage (MediaPipe)
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_confidence,
        )

        # --- Scan rapide des marges sûres
        min_left = float("inf")
        min_right = float("inf")
        min_top = float("inf")
        min_bottom = float("inf")
        frames_seen = 0
        frames_with_faces = 0
        total_faces = 0

        i = -1
        while True:
            i += 1
            ok = cap.grab()
            if not ok:
                break
            if detection_step > 1 and (i % detection_step) != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames_seen += 1

            # Détection sur image réduite pour la vitesse
            h0, w0 = frame.shape[:2]
            small_w = 320
            small_h = max(1, int(h0 * small_w / max(1, w0)))
            small = cv2.resize(frame, (small_w, small_h))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)
            if not results.detections:
                continue
            frames_with_faces += 1

            for det in results.detections:
                conf = float(det.score[0]) if det.score else 0.0
                if conf < min_confidence:
                    continue
                total_faces += 1
                bb = det.location_data.relative_bounding_box
                x = int(bb.xmin * small_w)
                y = int(bb.ymin * small_h)
                bw = int(bb.width * small_w)
                bh = int(bb.height * small_h)
                # Reprojeter en pleine résolution
                sx, sy = w0 / small_w, h0 / small_h
                x = int(x * sx); y = int(y * sy); bw = int(bw * sx); bh = int(bh * sy)

                left_margin = x
                right_margin = w0 - (x + bw)
                top_margin = y
                bottom_margin = h0 - (y + bh)
                if left_margin < min_left:   min_left = left_margin
                if right_margin < min_right: min_right = right_margin
                if top_margin < min_top:     min_top = top_margin
                if bottom_margin < min_bottom: bottom_margin = bottom_margin if False else min_bottom
                # the previous line had a typo; correct assignment:
                if bottom_margin < min_bottom:
                    min_bottom = bottom_margin

        # Décision de crop horizontal (symétrique)
        if min_left == float("inf") or min_right == float("inf"):
            # Aucun visage fiable sur l'échantillon → pas de crop
            crop_x = 0
            crop_w = W
            self.log("Aucun visage détecté de façon fiable → pas de crop horizontal")
        else:
            safe_left = max(0, int(min_left - margin_px))
            safe_right = max(0, int(min_right - margin_px))
            sym_crop = max(0, min(safe_left, safe_right))
            # Ne pas dépasser 40% de la largeur pour éviter un zoom excessif
            sym_crop = min(sym_crop, int(0.40 * W))
            crop_x = sym_crop
            crop_w = max(1, W - 2 * sym_crop)
            self.log(f"Crop horizontal: {sym_crop}px de chaque côté → zone {crop_w}x{H}")

        detect_rate = (frames_with_faces / max(1, frames_seen)) * 100.0
        self.log(f"Scan visages: frames analysées={frames_seen}, détection sur {detect_rate:.1f}% des frames, visages total={total_faces}")

        # --- Helpers d'assemblage
        def compose_blur_background(base_frame, inner_frame, target_size):
            tw, th = target_size
            fh, fw = base_frame.shape[:2]
            # BG cover
            s_bg = max(tw / fw, th / fh)
            bg = cv2.resize(base_frame, (int(fw * s_bg), int(fh * s_bg)))
            bh, bw = bg.shape[:2]
            x0 = max(0, (bw - tw) // 2)
            y0 = max(0, (bh - th) // 2)
            bg = bg[y0:y0 + th, x0:x0 + tw]
            bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=max(1, tw // 100))
            # Center inner
            ih, iw = inner_frame.shape[:2]
            canvas = bg.copy()
            x1 = (tw - iw) // 2
            y1 = (th - ih) // 2
            canvas[y1:y1 + ih, x1:x1 + iw] = inner_frame
            return canvas

        # --- Deuxième passe: rendu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        last_progress_ratio_logged = -1.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Extraire la bande horizontale croppée
            sub = frame[:, crop_x:crop_x + crop_w]
            # Mise à l'échelle pour remplir la largeur cible
            scale = ow / sub.shape[1]
            new_h = int(sub.shape[0] * scale)
            scaled = cv2.resize(sub, (ow, new_h))

            if new_h >= oh:
                # Recadrage vertical centré (pas de déformation)
                y0 = (new_h - oh) // 2
                current = scaled[y0:y0 + oh, :]
            else:
                # Letterbox vertical → combler avec fond flou
                current = compose_blur_background(frame, scaled, (ow, oh))

            out.write(current)

            if n_frames > 0:
                ratio = frame_idx / n_frames
                if ratio - last_progress_ratio_logged >= progress_log_step:
                    self.log(f"Smart-crop: {int(ratio*100):3d}%")
                    last_progress_ratio_logged = ratio

        cap.release()
        out.release()
        face_detector.close()
        self.log(f"Montage smart-crop enregistré dans {self.output_video}")
    """Handle the different video processing steps."""

    def __init__(self, log_callback, progress_callback, done_callback,):
        self.log = log_callback
        self.update_progress = progress_callback
        self.done_callback = done_callback
        self.output_video: str | None = None
        self.tmp_dir = tempfile.mkdtemp(prefix="yt_dl_")
        self.audio_path: str | None = None
        self.video_path: str | None = None
        self.title: str | None = None
        self.author: str | None = None
        self.description_tiktok: str | None = None
        self.hashtag: [str] = None


    def download_youtube_video(self, url: str) -> None:
        """Download a YouTube video and return title, video path and audio path."""
        yt = YouTube(url)
        self.log(f"Titre de la vidéo : {yt.title}")
        self.title = re.sub(r'[\\/*?:"<>|]', "_", yt.title)
        self.author = yt.author
        self.hashtag = re.findall(r"#\w+", yt.description)

        video_stream = (
            yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        if not video_stream:
            raise Exception("Aucun flux vidéo disponible.")

        audio_stream = (
            yt.streams.filter(adaptive=True, only_audio=True, file_extension="mp4")
            .order_by("abr")
            .desc()
            .first()
        )
        if not audio_stream:
            raise Exception("Aucun flux audio disponible.")

        self.video_path = video_stream.download(output_path=self.tmp_dir)
        self.audio_path = audio_stream.download(output_path=self.tmp_dir)

    def add_audio_to_video(self,video_without_audio: str,start_time: float,end_time: float,) -> str:
        """Merge a segment of audio into a video clip."""
        output_video = (
            video_without_audio.replace(".mp4", "_with_audio.mp4")
            if video_without_audio.endswith(".mp4")
            else f"{video_without_audio}.mp4"
        )

        audio_segment = "temp_audio_segment.m4a"
        extract_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.audio_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c",
            "copy",
            audio_segment,
        ]
        subprocess.run(extract_cmd, check=True)

        merge_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_without_audio,
            "-i",
            audio_segment,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_video,
        ]
        subprocess.run(merge_cmd, check=True)

        try:
            os.remove(audio_segment)
        except OSError:
            pass

        return output_video



    def _probe_duration(self, path: str) -> float:
        """Return media duration in seconds using ffprobe (fallback when CAP_PROP_FRAME_COUNT is 0)."""
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            return float((result.stdout or "0").strip())
        except Exception:
            return 0.0

    def detect_scenes_ffmpeg(self, path: str, threshold: float = 0.40) -> list[float]:
        """
        Detect scene changes using FFmpeg's 'scene' score.
        Returns a sorted list of times (in seconds) where a cut is detected.
        `threshold` higher -> fewer cuts. Typical values 0.35–0.50.
        """
        import re
        cmd = [
            "ffmpeg", "-hide_banner", "-i", path,
            "-filter:v", f"select=gt(scene,{threshold}),showinfo",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        times = []
        pattern = re.compile(r"pts_time:(\d+\.\d+).*scene:(\d+\.\d+)")
        for line in (proc.stderr or "").splitlines():
            m = pattern.search(line)
            if m:
                t = float(m.group(1))
                times.append(t)
        # Deduplicate & sort
        return sorted(set(times))

    def cut_into_clips(self) -> list[str]:
        """Cut the video into 61-second clips with audio."""
        self.log("Découpage en clips de 61 secondes...")

        output_clips = []
        downloads = get_downloads_dir()

        command_duration = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.video_path,
        ]
        result = subprocess.run(
            command_duration,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        total_duration = float(result.stdout.strip())
        segment_duration = 61

        start = 0
        index = 1
        while start < total_duration:
            clip_name = f"clip_{index:02d}.mp4"
            output_path = Path(self.tmp_dir) / clip_name
            command_cut = [
                "ffmpeg",
                "-y",
                "-ss",
                str(int(start)),
                "-i",
                str(self.output_video),
                "-t",
                str(segment_duration),
                "-c",
                "copy",
                str(output_path),
            ]
            subprocess.run(command_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            clip_start = start
            clip_end = min(start + segment_duration, total_duration)
            clips_dir = downloads / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            final_clip_path = clips_dir / clip_name
            output_with_audio = self.add_audio_to_video(str(output_path), clip_start, clip_end)
            shutil.move(output_with_audio, final_clip_path)
            output_clips.append(str(final_clip_path))
            index += 1
            start += segment_duration

        self.log(f"{len(output_clips)} clips générés dans {clips_dir}")
        return output_clips

    def add_subtitles_to_video(self,ass_filepath) -> None:
        """
        Add subtitles from an ASS file to a video using ffmpeg.
        """
        self.log("--- Ajout des subtitles ---")

        output_path = str(self.output_video).replace(".mp4", "_subtitled.mp4")

        command = [
            "ffmpeg",
            "-y",
            "-i", self.output_video,
            "-vf", f"subtitles={ass_filepath}",
            "-c:a", "copy",
            output_path,
        ]
        subprocess.run(command, check=True)
        self.output_video = output_path

    async def process(self, source: str, zoom_percent: int) -> None:
        self.log("\n--- Démarrage du traitement ---")
        try:
            self.update_progress(0)

            self.download_youtube_video(source)
            self.update_progress(1 / 5)

            # Détection des scènes en amont pour un log immédiat
            scene_threshold = 0.40
            total_duration = self._probe_duration(self.video_path)
            cuts = self.detect_scenes_ffmpeg(self.video_path, threshold=scene_threshold)
            scene_starts = [0.0] + [t for t in cuts if 0.0 < t < max(0.0, total_duration)]
            scene_ends = scene_starts[1:] + [total_duration]
            scenes = list(zip(scene_starts, scene_ends)) if total_duration > 0 else [(0.0, float("inf"))]
            short_lt10 = sum(1 for s, e in scenes if (e - s) < 10.0)
            mid_10_15 = sum(1 for s, e in scenes if 10.0 <= (e - s) < 15.0)
            long_ge15 = sum(1 for s, e in scenes if (e - s) >= 15.0)
            self.log(f"Plans détectés: {len(scenes)} ( <10s: {short_lt10} • 10–15s: {mid_10_15} • ≥15s: {long_ge15} )")

            # Lancement des tâches asynchrones
            ass_filepath = asyncio.create_task(utils.generate_styled_subtitles(self.audio_path))

            self.montage_smart_crop()
            self.update_progress(2 / 5)

            self.add_subtitles_to_video(await ass_filepath)
            self.update_progress(3 / 5)

            self.cut_into_clips()
            self.update_progress(4 / 5)

            self.log(f"Génération de la Description Tiktok... ")
            self.description_tiktok = f"{self.title} - {self.author} \n{' '.join(self.hashtag[:3])}"
            self.log(f"Description Tiktok : {self.description_tiktok} ")
            self.update_progress(5 / 5)

            self.log("Traitement terminé")
            time.sleep(5)
            self.update_progress(1)
            self.done_callback(self.output_video)
        except Exception as exc:
            self.log(f"Erreur: {exc}")
        finally:
            if self.video_path and os.path.exists(self.video_path):
                safe_rmtree(os.path.dirname(self.video_path))
            self.update_progress(0)
