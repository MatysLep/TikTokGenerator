import os
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
import math

import cv2
from pytubefix import YouTube
import mediapipe as mp
import utils
from utils import *

class VideoProcessor:
    """
    MVP minimal et non configurable.
    - Sortie fixe: 1080x1920 (9:16)
    - Confiance détection visage: 0.60
    - Marge de sécurité: 15 px
    - Pas d'options UI: aucune personnalisation
    - Découpe en clips de 61s avec audio
    """
    _OUT_W = 1080
    _OUT_H = 1920
    _MIN_CONF = 0.80
    _MARGIN_PX = 15
    _DETECTION_STEP = 5
    _CLIP_SECONDS = 61

    def __init__(self, log_callback, progress_callback, done_callback):
        # Callbacks
        self.update_progress_percentage = 0.0
        self.log = log_callback
        self.update_progress = progress_callback
        self.done_callback = done_callback

        # État / fichiers
        self.tmp_dir = tempfile.mkdtemp(prefix="yt_dl_")
        self.output_video: str | None = None
        self.audio_path: str | None = None
        self.video_path: str | None = None
        self.title: str | None = None
        self.author: str | None = None
        self.description_tiktok: str | None = None
        self.hashtag: list[str] = []
        self.clips: list[str] = []

    # ---------- IO ----------
    def download_youtube_video(self, url: str) -> None:
        """Télécharge une vidéo YouTube (flux adaptatifs) + audio AAC."""
        yt = YouTube(url)
        self.log(f"Titre de la vidéo : {yt.title}")
        self.title = re.sub(r'[\\/*?:"<>|]', "_", yt.title)
        self.author = yt.author
        self.hashtag = re.findall(r"#\w+", yt.description or "")

        video_stream = (
            yt.streams.filter(adaptive=True, only_video=True, file_extension="mp4")
            .order_by("resolution").desc().first()
        )
        if not video_stream:
            raise Exception("Aucun flux vidéo disponible.")

        audio_stream = (
            yt.streams.filter(adaptive=True, only_audio=True, file_extension="mp4")
            .order_by("abr").desc().first()
        )
        if not audio_stream:
            raise Exception("Aucun flux audio disponible.")

        self.video_path = video_stream.download(output_path=self.tmp_dir)
        self.audio_path = audio_stream.download(output_path=self.tmp_dir)

    def load_local_video(self, path: str) -> None:
        """Prépare un fichier vidéo local et extrait l'audio AAC temporaire."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path}")
        self.video_path = str(p)
        self.title = p.stem
        self.author = "Local file"
        self.hashtag = []

        audio_out = os.path.join(self.tmp_dir, f"{p.stem}.m4a")
        extract_cmd = ["ffmpeg", "-y", "-i", str(p), "-vn", "-acodec", "aac", audio_out]
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.audio_path = audio_out

    # ---------- Probing ----------
    def _probe_duration(self, path: str) -> float:
        """Durée (s) via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float((res.stdout or "0").strip())
        except Exception:
            return 0.0

    def detect_scenes_ffmpeg(self, path: str, threshold: float = 0.40) -> list[float]:
        """Détecte les coupes (scores 'scene') et retourne les temps (s)."""
        cmd = [
            "ffmpeg", "-hide_banner", "-i", path,
            "-filter:v", f"select=gt(scene,{threshold}),showinfo",
            "-f", "null", "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        times: list[float] = []
        pattern = re.compile(r"pts_time:(\d+\.\d+).*scene:(\d+\.\d+)")
        for line in (proc.stderr or "").splitlines():
            m = pattern.search(line)
            if m:
                times.append(float(m.group(1)))
        return sorted(set(times))

    # ---------- Smart crop (fixe) ----------
    def montage_smart_crop(self) -> None:
        """
        Crop horizontal symétrique maximisé sans couper les visages.
        - Remplit la largeur 1080; garde les proportions; fond flou si hauteur < 1920
        """
        self.log("Démarrage du montage smart-crop…")
        in_path = self.video_path
        assert in_path, "Chemin vidéo introuvable"

        cap = cv2.VideoCapture(in_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0:
            duration = self._probe_duration(in_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if duration > 0 and frame_count > 0:
                fps = max(1.0, frame_count / duration)
            else:
                fps = 25.0

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames == 0:
            n_frames = int(round(fps * max(0.0, self._probe_duration(in_path))))

        ow, oh = self._OUT_W, self._OUT_H

        base = Path(in_path).stem
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_video = os.path.join(self.tmp_dir, f"{base}_smartcrop.mp4")
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (ow, oh))

        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self._MIN_CONF,
        )

        # Scan marges sûres
        min_left = float("inf"); min_right = float("inf")
        min_top = float("inf");  min_bottom = float("inf")
        frames_seen = 0; frames_with_faces = 0; total_faces = 0

        i = -1
        while True:
            i += 1
            ok = cap.grab()
            if not ok:
                break
            if self._DETECTION_STEP > 1 and (i % self._DETECTION_STEP) != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames_seen += 1

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
                if conf < self._MIN_CONF:
                    continue
                total_faces += 1
                bb = det.location_data.relative_bounding_box
                x = int(bb.xmin * small_w)
                y = int(bb.ymin * small_h)
                bw = int(bb.width * small_w)
                bh = int(bb.height * small_h)
                # Reprojeter
                sx, sy = w0 / small_w, h0 / small_h
                x = int(x * sx); y = int(y * sy); bw = int(bw * sx); bh = int(bh * sy)

                left_margin   = x
                right_margin  = w0 - (x + bw)
                top_margin    = y
                bottom_margin = h0 - (y + bh)
                if left_margin   < min_left:   min_left = left_margin
                if right_margin  < min_right:  min_right = right_margin
                if top_margin    < min_top:    min_top = top_margin
                if bottom_margin < min_bottom: min_bottom = bottom_margin

        # Décision de crop horizontal
        if min_left == float("inf") or min_right == float("inf"):
            crop_x = 0; crop_w = W
            self.log("Aucun visage détecté de façon fiable → pas de crop horizontal")
        else:
            safe_left  = max(0, int(min_left  - self._MARGIN_PX))
            safe_right = max(0, int(min_right - self._MARGIN_PX))
            sym_crop = max(0, min(safe_left, safe_right))
            sym_crop = min(sym_crop, int(0.40 * W))  # garde-fou
            crop_x = sym_crop
            crop_w = max(1, W - 2 * sym_crop)
            self.log(f"Crop horizontal: {sym_crop}px de chaque côté → zone {crop_w}x{H}")

        detect_rate = (frames_with_faces / max(1, frames_seen)) * 100.0
        self.log(
            f"Scan visages: frames analysées={frames_seen}, "
            f"détection sur {detect_rate:.1f}% des frames, visages total={total_faces}"
        )

        def compose_blur_background(base_frame, inner_frame, target_size):
            tw, th = target_size
            fh, fw = base_frame.shape[:2]
            s_bg = max(tw / fw, th / fh)
            bg = cv2.resize(base_frame, (int(fw * s_bg), int(fh * s_bg)))
            bh, bw = bg.shape[:2]
            x0 = max(0, (bw - tw) // 2)
            y0 = max(0, (bh - th) // 2)
            bg = bg[y0:y0 + th, x0:x0 + tw]
            bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=max(1, tw // 100))
            ih, iw = inner_frame.shape[:2]
            canvas = bg.copy()
            x1 = (tw - iw) // 2
            y1 = (th - ih) // 2
            canvas[y1:y1 + ih, x1:x1 + iw] = inner_frame
            return canvas

        # Rendu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        last_ratio = -1.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            sub = frame[:, crop_x:crop_x + crop_w]
            scale = self._OUT_W / sub.shape[1]
            new_h = int(sub.shape[0] * scale)
            scaled = cv2.resize(sub, (self._OUT_W, new_h))

            if new_h >= self._OUT_H:
                y0 = (new_h - self._OUT_H) // 2
                current = scaled[y0:y0 + self._OUT_H, :]
            else:
                current = compose_blur_background(frame, scaled, (self._OUT_W, self._OUT_H))

            out.write(current)

            if n_frames > 0:
                ratio = frame_idx / n_frames
                if ratio - last_ratio >= 0.10:
                    self.increment_progress(0.05)
                    last_ratio = ratio

        cap.release()
        out.release()
        face_detector.close()
        self.log(f"Montage smart-crop enregistré dans {self.output_video}")

    # ---------- Subtitles & audio ----------
    def add_subtitles_to_video(self, ass_filepath: str) -> None:
        """Brûle des sous-titres ASS dans la vidéo (ffmpeg)."""
        self.log("--- Ajout des subtitles ---")
        output_path = str(self.output_video).replace(".mp4", "_subtitled.mp4")
        command = [
            "ffmpeg", "-y", "-i", self.output_video,
            "-vf", f"subtitles={ass_filepath}", "-c:a", "copy", output_path,
        ]
        subprocess.run(command, check=True)
        self.output_video = output_path

    def add_audio_to_video(self, video_without_audio: str, start_time: float, end_time: float) -> str:
        """Fusionne un segment audio AAC dans un clip vidéo en privilégiant le stream copy.
        - Extraction audio par segment via `-ss/-to` **avant** l'entrée (seek rapide)
        - Mux vidéo+audio en copy (sans ré-encodage) et `-shortest` (évite padding infini)
        - `+faststart` pour une lecture web fluide
        """
        output_video = (
            video_without_audio[:-4] + "_with_audio.mp4"
            if video_without_audio.endswith(".mp4") else f"{video_without_audio}_with_audio.mp4"
        )

        # Fichier audio segmenté unique (évite collisions et I/O inutiles)
        audio_segment = os.path.join(
            self.tmp_dir, f"aud_{int(start_time)}_{int(end_time)}.m4a"
        )

        # Extraction audio rapide en copy
        extract_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-to", f"{end_time:.3f}",
            "-i", self.audio_path,
            "-c", "copy",
            "-movflags", "+faststart",
            audio_segment,
        ]
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Mux sans ré-encodage, on coupe au plus court pour éviter toute dérive de durée
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", video_without_audio, "-i", audio_segment,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            output_video,
        ]
        subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            os.remove(audio_segment)
        except OSError:
            pass

        return output_video

    def cut_into_clips(self) -> list[str]:
        """Découpe la vidéo finale en clips de 61s avec audio (rapide, stream copy)."""
        if not self.output_video or not os.path.exists(self.output_video):
            self.log("Avertissement: aucune vidéo de sortie trouvée pour le découpage.")
            return []

        downloads = get_downloads_dir()
        clips_dir = downloads / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Durée totale (sur la vidéo de sortie)
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", self.output_video,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            total_duration = float((res.stdout or "0").strip())
        except Exception:
            total_duration = 0.0

        if total_duration <= 0.0:
            self.log("Durée inconnue (0s) → aucun découpage effectué.")
            return []

        self.log("Découpage en clips de 61 secondes…")

        segment_duration = float(self._CLIP_SECONDS)
        output_clips: list[str] = []
        start = 0.0
        index = 1

        while start < total_duration - 0.001:
            clip_name = f"clip_{index:02d}.mp4"
            tmp_clip_path = Path(self.tmp_dir) / clip_name

            # Découpe vidéo ultrarapide (seek avant l'entrée + copy)
            cut_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(self.output_video),
                "-t", f"{min(segment_duration, total_duration - start):.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-reset_timestamps", "1",
                str(tmp_clip_path),
            ]
            subprocess.run(cut_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            clip_start = start
            clip_end = min(start + segment_duration, total_duration)
            final_clip_path = clips_dir / clip_name

            # Ajout audio (copy + shortest) puis déplacement vers le dossier final
            out_with_audio = self.add_audio_to_video(str(tmp_clip_path), clip_start, clip_end)
            shutil.move(out_with_audio, final_clip_path)

            output_clips.append(str(final_clip_path))
            self.log(f"   ✓ Écrit: {final_clip_path.name}")

            # Petite incrémentation de progression visible
            self.increment_progress(0.02)

            index += 1
            start += segment_duration

        self.log(f"{len(output_clips)} clips générés dans {clips_dir}")
        self.clips = output_clips
        return output_clips

    def _persist_output(self) -> str | None:
        """Déplace la vidéo finale vers ~/Downloads/final et retourne le chemin persistant.
        Évite que Streamlit pointe vers un fichier supprimé lors du nettoyage du répertoire temporaire.
        """
        downloads = get_downloads_dir()
        out_dir = downloads / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.output_video:
            return None
        src = Path(self.output_video)
        if not src.exists():
            return None
        base = re.sub(r'[\\/*?:"<>|]', "_", (self.title or src.stem))
        dest = out_dir / f"{base}.mp4"
        i = 1
        while dest.exists():
            dest = out_dir / f"{base}_{i:02d}.mp4"
            i += 1
        shutil.move(str(src), str(dest))
        self.output_video = str(dest)
        return self.output_video

    def increment_progress(self, step: float) -> None:
        """
        Incrémente la progression et déclenche le callback.
        :param step: valeur entre 0 et 1 (ex: 0.05 = +5%)
        """
        self.update_progress_percentage = min(1.0, self.update_progress_percentage + step)
        self.update_progress(self.update_progress_percentage)

    # ---------- Orchestration ----------
    async def process(self, source: str) -> None:
        """Pipeline fixe sans paramètres personnalisables."""
        self.log("\n--- Démarrage du traitement ---")
        try:
            self.increment_progress(0.0)

            # 1) Source
            if source.lower().startswith(("http://", "https://")):
                self.download_youtube_video(source)
            else:
                self.load_local_video(source)
            self.increment_progress(0.10)

            # 2) Détection des scènes (info)
            total_duration = self._probe_duration(self.video_path)
            cuts = self.detect_scenes_ffmpeg(self.video_path, threshold=0.40)
            scene_starts = [0.0] + [t for t in cuts if 0.0 < t < max(0.0, total_duration)]
            scene_ends = scene_starts[1:] + [total_duration]
            scenes = list(zip(scene_starts, scene_ends)) if total_duration > 0 else [(0.0, float("inf"))]
            short_lt10 = sum(1 for s, e in scenes if (e - s) < 10.0)
            mid_10_15 = sum(1 for s, e in scenes if 10.0 <= (e - s) < 15.0)
            long_ge15 = sum(1 for s, e in scenes if (e - s) >= 15.0)
            self.log(f"Plans détectés: {len(scenes)} ( <10s: {short_lt10} • 10–15s: {mid_10_15} • ≥15s: {long_ge15} )")

            # 3) Smart crop (fixe)
            self.montage_smart_crop()
            self.increment_progress(0.10)

            # 4) Sous-titres (style par défaut dans utils)
            try:
                ass_task = asyncio.create_task(utils.generate_styled_subtitles(self.audio_path))
            except TypeError:
                ass_task = asyncio.create_task(utils.generate_styled_subtitles(self.audio_path))
            ass_filepath = await ass_task
            self.add_subtitles_to_video(ass_filepath)
            self.increment_progress(0.10)

            # 5) Découpe en clips
            clips = self.cut_into_clips()
            self.clips = clips
            self.increment_progress(0.10)

            # 5bis) Déplacer la vidéo finale dans un emplacement persistant (Downloads/final)
            self._persist_output()

            # 6) Description minimale
            self.description_tiktok = f"{self.title} - {self.author} \n{' '.join(self.hashtag[:3])}"
            self.log(f"Description Tiktok : {self.description_tiktok} ")
            self.increment_progress(0.10)

            self.log("Traitement terminé")
            if self.output_video:
                self.done_callback(self.output_video)
        except Exception as exc:
            self.log(f"Erreur: {exc}")
        finally:
            # Nettoyage strict: ne supprimer que le répertoire temporaire alloué par l'instance
            if hasattr(self, "tmp_dir") and self.tmp_dir and os.path.isdir(self.tmp_dir):
                safe_rmtree(self.tmp_dir)
            self.update_progress(0)