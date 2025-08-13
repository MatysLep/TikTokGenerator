# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Interface Streamlit minimaliste pour piloter VideoProcessor
# MVP : URL YouTube OU fichier local → lancer → logs + progression → résultat
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Callable, Tuple

import streamlit as st

from video_processor import VideoProcessor

# ─────────────────────────────────────────────────────────────────────────────
# État & callbacks
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    """Initialise l'état Streamlit au premier run."""
    defaults = {
        "logs": [],
        "progress": 0.0,
        "output_video": None,
        "clips": [],
        "title": "",
        "author": "",
        "description_tiktok": "",
        "hashtags_str": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def make_callbacks(
    log_box: st.delta_generator.DeltaGenerator,
    progress_bar: st.delta_generator.DeltaGenerator,
) -> Tuple[Callable[[str], None], Callable[[float], None], Callable[[str], None]]:
    """
    Fabrique les callbacks requis par VideoProcessor avec MAJ de l'UI.

    - log_callback(msg): ajoute une ligne de log.
    - progress_callback(p): p ∈ [0,1], met à jour la barre.
    - done_callback(path): appelé en fin de traitement (sécurité : on lit aussi l'état objet).
    """

    def log_callback(msg: str) -> None:
        st.session_state.logs.append(str(msg))
        # Affichage instantané
        log_box.code("\n".join(st.session_state.logs[-500:]), language="bash")

    def progress_callback(p: float) -> None:
        p = max(0.0, min(1.0, float(p)))
        st.session_state.progress = p
        progress_bar.progress(p)

    def done_callback(path: str) -> None:
        st.session_state.output_video = path

    return log_callback, progress_callback, done_callback


# ─────────────────────────────────────────────────────────────────────────────
# Exécution du pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(source: str, callbacks) -> None:
    """
    Exécute le pipeline asynchrone de VideoProcessor dans ce thread.

    Remplit st.session_state avec :
      - output_video, clips, title, author, description_tiktok, hashtags_str.
    """
    log_cb, prog_cb, done_cb = callbacks
    vp = VideoProcessor(log_cb, prog_cb, done_cb)

    # Exécuter l'asynchrone proprement dans Streamlit
    asyncio.run(vp.process(source))

    # Récupérer l'état final de manière déterministe (au cas où done_cb ne serait pas invoqué)
    st.session_state.output_video = vp.output_video
    st.session_state.clips = vp.clips or []
    st.session_state.title = vp.title or ""
    st.session_state.author = vp.author or ""
    st.session_state.description_tiktok = vp.description_tiktok or ""
    st.session_state.hashtags_str = " ".join(vp.hashtag[:10] if getattr(vp, "hashtag", []) else [])


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def render_header() -> None:
    st.set_page_config(page_title="TikTok Clip Generator", page_icon="🎬", layout="centered")
    st.markdown(
        """
        <style>
        .block-container { max-width: 840px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🎬 TikTok Clip Generator")
    st.caption("Entrer un lien YouTube **ou** choisir un fichier local, puis lancer le traitement.")


def render_input_section() -> str | None:
    with st.container(border=True):
        mode = st.radio("Source", ["Lien YouTube", "Fichier local"], horizontal=True, index=0)
        if mode == "Lien YouTube":
            url = st.text_input("Lien YouTube", placeholder="https://www.youtube.com/watch?v=…")
            return url.strip() if url else None
        else:
            up = st.file_uploader("Fichier vidéo (mp4/mov/mkv)", type=["mp4", "mov", "mkv"])
            if up is None:
                return None
            # Sauvegarder en fichier temporaire (passage chemin → backend)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tmp.write(up.read())
            tmp.flush()
            return tmp.name


def render_controls_and_run(source: str | None, callbacks) -> None:
    # Bouton lancer
    cols = st.columns([1, 1, 2])
    run_clicked = cols[0].button("🚀 Lancer", use_container_width=True, disabled=not bool(source))
    reset_clicked = cols[1].button("♻️ Réinitialiser", use_container_width=True)

    if reset_clicked:
        # Reset doux
        for k in ("logs", "progress", "output_video", "clips", "title", "author", "description_tiktok", "hashtags_str"):
            st.session_state[k] = [] if k == "logs" else (0.0 if k == "progress" else None if "video" in k else "")
        st.success("État réinitialisé.")
        st.stop()

    if run_clicked and source:
        # Purger logs/état avant run
        st.session_state.logs = []
        st.session_state.progress = 0.0
        st.session_state.output_video = None
        st.session_state.clips = []
        st.session_state.description_tiktok = ""
        st.session_state.hashtags_str = ""
        st.session_state.title = ""
        st.session_state.author = ""

        with st.spinner("Traitement en cours…"):
            run_pipeline(source, callbacks)

        if st.session_state.output_video:
            st.success("Traitement terminé ✅")
        else:
            st.error("Le traitement s'est terminé sans vidéo de sortie. Consulte les logs.")


def render_progress_and_logs(placeholders) -> None:
    progress_bar, log_box = placeholders
    # Afficher état courant (utile si on rafraîchit ou si logs existent déjà)
    progress_bar.progress(st.session_state.progress)
    if st.session_state.logs:
        log_box.code("\n".join(st.session_state.logs[-500:]), language="bash")


def render_results() -> None:
    if not st.session_state.output_video:
        return

    st.subheader("🎞️ Résultat")
    if st.session_state.output_video and Path(st.session_state.output_video).exists():
        st.video(st.session_state.output_video)  # Aperçu du mp4 final (optionnel)
    else:
        st.warning("Le fichier vidéo final n'est plus accessible (probablement nettoyé du répertoire temporaire). Chemin :")
        st.code(st.session_state.output_video or "", language="bash")

    st.markdown("**Chemin vidéo finale :**")
    st.code(st.session_state.output_video, language="bash")

    if st.session_state.clips:
        st.markdown("**Clips générés (61s) :**")
        for p in st.session_state.clips:
            st.code(p, language="bash")

    # Zone “copier/coller” pour titre & hashtags
    st.subheader("📝 Métadonnées prêtes à copier")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Titre", value=st.session_state.title or "", help="Titre original")
        st.text_input("Auteur", value=st.session_state.author or "", help="Auteur")
    with col2:
        st.text_area("Description TikTok", value=st.session_state.description_tiktok or "", height=90)
        st.text_area("Hashtags", value=st.session_state.hashtags_str or "", height=90)


# ─────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    render_header()

    # Placeholders réutilisables (pour mises à jour live via callbacks)
    progress_bar = st.empty()
    log_box = st.empty()

    # Callbacks branchés sur ces placeholders
    callbacks = make_callbacks(log_box, progress_bar)

    # Saisie source
    source = render_input_section()

    # Contrôles + exécution
    render_controls_and_run(source, callbacks)

    # Rappels d'état & logs
    render_progress_and_logs((progress_bar, log_box))

    # Résultats
    render_results()


if __name__ == "__main__":
    main()