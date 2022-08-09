"""
Description taken from the official website of wenetspeech
(https://wenet-e2e.github.io/WenetSpeech/)

We release a 10000+ hours multi-domain transcribed Mandarin Speech Corpus
collected from YouTube and Podcast. Optical character recognition (OCR) and
automatic speech recognition (ASR) techniques are adopted to label each YouTube
and Podcast recording, respectively. To improve the quality of the corpus,
we use a novel end-to-end label error detection method to further validate and
filter the data.

See https://github.com/wenet-e2e/WenetSpeech for more details about WenetSpeech
"""

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    compute_num_samples,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations

import scipy.io.wavfile as wav

def read_kalditype_data(corpus_dir: Pathlike):
    corpus_dir = Path(corpus_dir)
    wavscp_file = corpus_dir / "wav.scp"
    segments_file = corpus_dir / "segments"
    u2s_file = corpus_dir / "utt2spk"
    s2u_file = corpus_dir / "spk2utt"
    text_file = corpus_dir / "text"

    aid2infos = defaultdict(dict)

    with open(wavscp_file, 'r') as fw, \
        open(segments_file, 'r') as fseg, \
        open(u2s_file, 'r') as fu2s, \
        open(text_file, 'r') as ft:
        for line in tqdm(fw.readlines(), desc="Read wav.scp..."):
            try:
                aid, audio_path = line.strip().split(" ", 1)
            except Exception as e:
                continue
            aid2infos[aid]["path"] = audio_path
            aid2infos[aid]["segments"] = []
        
        sid2text = {}
        for line in tqdm(ft.readlines(), desc="Read text...."):
            try:
                sid, text = line.strip().split(" ", 1)
            except Exception as e:
                continue
            sid2text[sid] = text
        
        sid2spk = {}
        for line in tqdm(fu2s.readlines(), desc="Read utt2spk..."):
            try:
                sid, spk = line.strip().split(" ", 1)
            except Exception as e:
                continue
            sid2spk[sid] = spk

        for line in tqdm(fseg.readlines(), desc="Reading Segments..."):
            try:
                sid, aid, start, end = line.strip().split(" ")
            except Exception as e:
                continue
            sid_info = {}
            sid_info["sid"] = sid
            sid_info["begin_time"] = float(start)
            sid_info["end_time"] = float(end)
            sid_info["text"] = sid2text[sid]
            sid_info["speaker"] = sid2spk[sid]
            aid2infos[aid]["segments"].append(sid_info)

    audios = []
    for aid, aid_info in aid2infos.items():
        audio = {"aid": aid, "path": aid_info["path"], "segments": aid_info["segments"]}
        audios.append(audio)
    return audios


def prepare_kalditype_speech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    subset_name: str = "dev",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: Which parts of dataset to prepare, all for all the
                          parts.
    :param output_dir: Pathlike, the path where to write the manifests.
    :num_jobs Number of workers to extract manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with
             the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {"recordings": [], "supervisions": []}
    audios = read_kalditype_data(corpus_dir)
    print(f" => Start to do the job with {num_jobs} threads...")
    with ProcessPoolExecutor(num_jobs) as ex:
        for recording, segments in tqdm(
            ex.map(
                parse_utterance,
                audios, 
                repeat(corpus_dir),
            ), desc="kaldi-type to k2-type..."
        ):
            if segments:
                manifests["recordings"].append(recording)
                manifests["supervisions"].extend(segments)

    recordings, supervisions = fix_manifests(
        recordings=RecordingSet.from_recordings(manifests["recordings"]),
        supervisions=SupervisionSet.from_segments(manifests["supervisions"]),
    )
    validate_recordings_and_supervisions(
        recordings=recordings, supervisions=supervisions
    )

    if output_dir is not None:
        supervisions.to_file(
            output_dir / f"so_kaldi_supervisions_{subset_name}.jsonl.gz"
        )
        recordings.to_file(output_dir / f"so_kaldi_recordings_{subset_name}.jsonl.gz")

    manifests = {
        "recordings": recordings,
        "supervisions": supervisions,
    }

    return manifests


def parse_utterance(
    audio: Any, root_path: Path
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:
    # sampling_rate = 16000
    sampling_rate, data = wav.read(audio["path"])
    audio["duration"] = len(data) / sampling_rate
    segments = []
    recording = Recording(
        id=audio["aid"],
        sources=[
            AudioSource(
                type="file",
                channels=[0],
                source=str(audio["path"]),
            )
        ],
        num_samples=compute_num_samples(
            duration=audio["duration"], sampling_rate=sampling_rate
        ),
        sampling_rate=sampling_rate,
        duration=audio["duration"],
    )
    if len(data.shape) != 1:
        return recording, segments
    recording = recording.resample(16000)
    for seg in audio["segments"]:
        segment = SupervisionSegment(
            id=seg["sid"],
            recording_id=audio["aid"],
            start=seg["begin_time"],
            duration=add_durations(
                seg["end_time"], -seg["begin_time"], sampling_rate=sampling_rate
            ),
            language="Chinese",
            speaker=seg["speaker"],
            text=seg["text"].strip()
        )
        segments.append(segment)
    return recording, segments
