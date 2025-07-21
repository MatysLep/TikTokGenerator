from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
from pytubefix import YouTube

from utils import get_downloads_dir, safe_rmtree

# Reduce TensorFlow logging from MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class VideoProcessor:
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


    def download_youtube_video(self, url: str) -> None:
        """Download a YouTube video and return title, video path and audio path."""
        yt = YouTube(url)
        self.log(f"Titre de la vidéo : {yt.title}")
        self.title = re.sub(r'[\\/*?:"<>|]', "_", yt.title)

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

    def add_audio_to_video(
        self,
        video_without_audio: str,
        start_time: float,
        end_time: float,
    ) -> str:
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

    def center_on_speaker(self, zoom_percent: int) -> str:
        """Center video on the detected speaker."""
        self.log("Centrage de la vidéo sur le locuteur ...")

        downloads_dir = get_downloads_dir()
        self.output_video = downloads_dir / f"{Path(self.video_path).stem}_centered.mp4"

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_ratio = 9 / 16
        out_w, out_h = width, int(width / target_ratio)
        if out_h > height:
            out_h = height
            out_w = int(height * target_ratio)

        writer = cv2.VideoWriter(str(self.output_video), fourcc, fps, (out_w, out_h))

        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        tolerance = 50
        smooth_factor = 0.2

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
                    if abs(cx_new - tx) > tolerance or abs(cy_new - ty) > tolerance:
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

        self.log(f"Vidéo centrée enregistrée dans {self.output_video}")
        return str(self.output_video)

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
                "-i",
                str(self.output_video),
                "-ss",
                str(int(start)),
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

    def process(self, source: str, zoom_percent: int) -> None:
        self.log("\n--- Démarrage du traitement ---")
        try:
            self.update_progress(0)

            self.download_youtube_video(source)
            self.update_progress(1 / 3)

            centered = self.center_on_speaker(zoom_percent)
            self.update_progress(2 / 3)

            self.cut_into_clips()
            self.update_progress(1)

            self.log("Traitement terminé")
            self.done_callback(centered)
        except Exception as exc:
            self.log(f"Erreur: {exc}")
        finally:
            if self.video_path and os.path.exists(self.video_path):
                safe_rmtree(os.path.dirname(self.video_path))
            self.update_progress(0)
