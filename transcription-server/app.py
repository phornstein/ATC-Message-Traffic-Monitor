"""ATC Voice Transcription Server - Processes live ATC audio and indexes to Elasticsearch."""

import os
import re
import gc
import queue
import logging
import threading
from datetime import datetime, timedelta

import torch
import numpy as np
import librosa
import requests
from flask import Flask, jsonify, send_file, abort
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load environment and configure PyTorch
load_dotenv()
os.environ.update({
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'PYTORCH_ENABLE_MPS_FALLBACK': '1',
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0'
})

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    RECORDINGS_DIR = os.getenv('RECORDINGS_DIR', 'recordings')
    CHUNK_DURATION = int(os.getenv('CHUNK_DURATION', 30))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 8192))
    MAX_RECORDINGS = int(os.getenv('MAX_RECORDINGS', 1000))
    OVERLAP_DURATION = int(os.getenv('OVERLAP_DURATION', 5))

    ES_ENDPOINT = os.getenv('ELASTICSEARCH_ENDPOINT')
    ES_API_KEY = os.getenv('ELASTICSEARCH_API_KEY')
    ES_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'atc-transcriptions')
    ELSER_MODEL_ID = os.getenv('ELSER_MODEL_ID', 'elser_2')

    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'jlvdoorn/whisper-large-v3-atco2-asr')
    VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', '0.02'))
    MIN_SILENCE_DURATION = float(os.getenv('MIN_SILENCE_DURATION', '0.5'))
    MIN_SPEECH_DURATION = float(os.getenv('MIN_SPEECH_DURATION', '0.3'))

    ATC_PLAYLIST_URL = os.getenv('ATC_PLAYLIST_URL', 'https://www.liveatc.net/play/korf1.pls')

os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)

# Flask app and global state
app = Flask(__name__)
audio_queue = queue.Queue(maxsize=100)
transcription_queue = queue.Queue(maxsize=50)
model = None
processor = None
es_client = None

# ATC patterns
ATC_VERBS = {"contact", "clear", "cleared", "climb", "descend", "maintain", "turn", "proceed", "hold", "taxi", "line", "wait", "expect"}
ATC_VERB_REGEX = re.compile(r"\b(" + "|".join(sorted(ATC_VERBS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
CALLSIGN_REGEX = re.compile(
    r"\b(?:(?!CONTACT\b|CLEAR\b|CLEARED\b|CLIMB\b|DESCEND\b|MAINTAIN\b|TURN\b|PROCEED\b|HOLD\b|TAXI\b|LINE\b|WAIT\b|EXPECT\b)"
    r"[A-Z]{3,15}\s+\d{2,4}|[A-Z]{2,3}\d{2,4}|(?:N|C|G|D|F|I|EC|JA|VH|ZS|OE|HB)(?=[A-Z0-9]*\d)[A-Z0-9]{2,6})\b",
    re.IGNORECASE
)


def remove_repetitions(text):
    """Remove Whisper hallucination repetitions from text."""
    if not text or len(text.split()) <= 3:
        return text

    words = text.split()

    # Remove excessive consecutive word repetitions (>2)
    cleaned = []
    i = 0
    while i < len(words):
        repeat_count = 1
        while i + repeat_count < len(words) and words[i + repeat_count] == words[i]:
            repeat_count += 1
        cleaned.extend([words[i]] * min(repeat_count, 2))
        i += repeat_count

    # Remove duplicate phrases (2-8 words)
    words = cleaned
    changed = True
    while changed:
        changed = False
        for phrase_len in range(8, 1, -1):
            if len(words) < phrase_len * 2:
                continue
            new_words = []
            i = 0
            while i < len(words):
                if i + phrase_len * 2 <= len(words) and words[i:i+phrase_len] == words[i+phrase_len:i+phrase_len*2]:
                    new_words.extend(words[i:i+phrase_len])
                    i += phrase_len * 2
                    changed = True
                else:
                    new_words.append(words[i])
                    i += 1
            words = new_words
            if changed:
                break

    # Remove trailing garbage
    garbage = {'the', 'to', 'thank', 'you', 'three', 'a', 'and', 'or'}
    while len(words) > 3 and words[-1].lower() in garbage and len(words) >= 2 and words[-2].lower() in garbage:
        words.pop()

    return ' '.join(words)


def detect_speech_segments(audio, sr):
    """Detect speech segments using energy-based VAD."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy = energy / (np.max(energy) + 1e-8)
    is_speech = energy > Config.VAD_THRESHOLD

    # Convert to time segments
    segments = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            start_time = start_frame * hop_length / sr
            end_time = i * hop_length / sr
            if (end_time - start_time) >= Config.MIN_SPEECH_DURATION:
                segments.append((start_time, end_time))
            in_speech = False

    if in_speech:
        start_time = start_frame * hop_length / sr
        if (len(audio) / sr - start_time) >= Config.MIN_SPEECH_DURATION:
            segments.append((start_time, len(audio) / sr))

    # Merge close segments
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] < Config.MIN_SILENCE_DURATION:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def extract_callsign(text):
    """Extract callsign from text."""
    match = CALLSIGN_REGEX.search(text)
    if match and not match.group(0).isalpha():
        return match.group(0)
    return None


def characterize_speaker(text):
    """Determine speaker type and callsign from text."""
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if parts:
        cs = extract_callsign(parts[0])
        if cs and cs.lower() not in ATC_VERBS:
            return "Aircraft", cs
        cs = extract_callsign(parts[-1])
        if cs and cs.lower() not in ATC_VERBS:
            return "Tower", cs
    return "Unknown", None


def cleanup_old_recordings():
    """Remove old recordings to save disk space."""
    try:
        if not os.path.exists(Config.RECORDINGS_DIR):
            return
        recordings = [(os.path.join(Config.RECORDINGS_DIR, f), os.path.getmtime(os.path.join(Config.RECORDINGS_DIR, f)))
                      for f in os.listdir(Config.RECORDINGS_DIR) if f.endswith('.mp3')]
        recordings.sort(key=lambda x: x[1])
        for filepath, _ in recordings[:max(0, len(recordings) - Config.MAX_RECORDINGS)]:
            os.remove(filepath)
            logger.info(f"Removed old recording: {os.path.basename(filepath)}")
    except Exception as e:
        logger.error(f"Error cleaning up recordings: {e}")


def setup_elasticsearch():
    """Initialize Elasticsearch client and setup index."""
    global es_client

    if not Config.ES_ENDPOINT or not Config.ES_API_KEY:
        logger.error("Elasticsearch endpoint or API key not configured")
        return False

    try:
        es_client = Elasticsearch(Config.ES_ENDPOINT, api_key=Config.ES_API_KEY, verify_certs=False)
        if not es_client.ping():
            logger.error("Could not connect to Elasticsearch")
            return False

        logger.info("Successfully connected to Elasticsearch")

        # Setup pipeline
        pipeline_name = f"{Config.ES_INDEX}-elser-pipeline"
        pipeline_body = {
            "description": "ELSER semantic text pipeline for ATC transcriptions",
            "processors": [{
                "inference": {
                    "model_id": Config.ELSER_MODEL_ID,
                    "input_output": [{"input_field": "message", "output_field": "message_semantic"}]
                }
            }]
        }

        try:
            es_client.ingest.get_pipeline(id=pipeline_name)
            logger.info(f"Pipeline '{pipeline_name}' already exists")
        except:
            es_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_body)
            logger.info(f"Created ingest pipeline '{pipeline_name}'")

        # Setup index
        if not es_client.indices.exists(index=Config.ES_INDEX):
            es_client.indices.create(index=Config.ES_INDEX, body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 1, "default_pipeline": pipeline_name},
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "start_time": {"type": "date"},
                        "end_time": {"type": "date"},
                        "speaker": {"type": "keyword"},
                        "callsign": {"type": "keyword"},
                        "message": {"type": "text"},
                        "message_semantic": {"type": "rank_features", "inference_id": Config.ELSER_MODEL_ID},
                        "audio_file": {"type": "keyword"},
                        "instructions": {"type": "keyword"},
                        "metadata": {"properties": {
                            "model": {"type": "keyword"},
                            "processing_time": {"type": "date"},
                            "segment_start": {"type": "float"},
                            "segment_end": {"type": "float"},
                            "stream_title": {"type": "text"}
                        }}
                    }
                }
            })
            logger.info(f"Created index '{Config.ES_INDEX}'")
        else:
            logger.info(f"Index '{Config.ES_INDEX}' already exists")

        return True
    except Exception as e:
        logger.error(f"Error setting up Elasticsearch: {e}")
        return False


def index_to_elasticsearch(data):
    """Index transcription data to Elasticsearch."""
    if not es_client:
        return
    try:
        data['@timestamp'] = datetime.now().isoformat()
        response = es_client.index(index=Config.ES_INDEX, document=data)
        logger.info(f"Indexed document: {response['_id']}")
    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {e}")


def thread_stream_reader():
    """Read from live ATC stream."""
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.liveatc.net/'}
    logger.info("Starting stream reader...")

    try:
        if Config.ATC_PLAYLIST_URL.endswith('.pls'):
            logger.info(f"Fetching PLS playlist from {Config.ATC_PLAYLIST_URL}")
            pls_response = requests.get(Config.ATC_PLAYLIST_URL, timeout=10, headers=headers)
            stream_urls = re.findall(r'File1=(.+?)(?:\n|$)', pls_response.text)
            stream_title = re.findall(r'Title1=(.+?)(?:\n|$)', pls_response.text)

            if not stream_urls:
                logger.error("Could not find stream URL in PLS")
                return

            audio_url = stream_urls[0].strip()
            logger.info(f"Found stream: {audio_url}")
        else:
            logger.info(f"Using direct stream URL: {Config.ATC_PLAYLIST_URL}")
            stream_urls = [Config.ATC_PLAYLIST_URL]
            stream_title = Config.ATC_PLAYLIST_URL.split('/')[-1]

        if not stream_urls:
            logger.error("Could not find stream URL")
            return

        audio_url = stream_urls[0].strip()
        logger.info(f"Found stream: {audio_url}")

        response = requests.get(audio_url, stream=True, timeout=None, headers=headers)
        response.raise_for_status()

        bitrate_kbps = int(response.headers.get('icy-br', '32'))
        bytes_per_second = (bitrate_kbps * 1000) / 8
        chunk_size_bytes = int(Config.CHUNK_DURATION * bytes_per_second)
        overlap_size_bytes = int(Config.OVERLAP_DURATION * bytes_per_second)

        logger.info(f"Bitrate: {bitrate_kbps} kbps | Chunk: {chunk_size_bytes} bytes | Overlap: {overlap_size_bytes} bytes")

        chunk_number = 0
        buffer = b""
        overlap_buffer = b""

        for data in response.iter_content(chunk_size=Config.CHUNK_SIZE):
            if data:
                buffer += data
                if len(buffer) >= chunk_size_bytes:
                    chunk_number += 1
                    timestamp = datetime.now()
                    filename = os.path.join(Config.RECORDINGS_DIR, f"atc_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.mp3")

                    with open(filename, 'wb') as f:
                        f.write(overlap_buffer + buffer[:chunk_size_bytes])

                    audio_queue.put({
                        'chunk_num': chunk_number,
                        'filename': filename,
                        'timestamp': timestamp.timestamp(),
                        'stream_title': stream_title[0] if stream_title else "Unknown"
                    })

                    logger.info(f"Queued chunk {chunk_number}")
                    overlap_buffer = buffer[max(0, chunk_size_bytes - overlap_size_bytes):chunk_size_bytes]
                    buffer = buffer[chunk_size_bytes:]

    except KeyboardInterrupt:
        logger.info("Stream interrupted")
    except Exception as e:
        logger.error(f"Stream error: {type(e).__name__}: {e}")


def thread_transcriber():
    """Transcribe audio chunks."""
    global model, processor

    logger.info(f"Loading Whisper model: {Config.WHISPER_MODEL}")

    try:
        torch.set_grad_enabled(False)
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)

        processor = AutoProcessor.from_pretrained(Config.WHISPER_MODEL)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(Config.WHISPER_MODEL).to("cpu").eval()
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    while True:
        try:
            audio_item = audio_queue.get(timeout=5)
        except queue.Empty:
            continue

        if audio_item is None:
            break

        chunk_num = audio_item['chunk_num']
        filename = audio_item['filename']
        recording_time = datetime.fromtimestamp(audio_item['timestamp'])

        logger.info(f"Transcribing chunk {chunk_num}...")

        try:
            audio_array, _ = librosa.load(filename, sr=16000, mono=True)
            segments = detect_speech_segments(audio_array, 16000)

            if not segments:
                logger.info(f"Chunk {chunk_num}: No speech detected")
                continue

            logger.info(f"Chunk {chunk_num}: {len(segments)} segment(s)")

            for seg_idx, (start_time, end_time) in enumerate(segments):
                segment_audio = audio_array[int(start_time * 16000):int(end_time * 16000)]
                if len(segment_audio) < 16000 * 0.3:
                    continue

                inputs = processor(segment_audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = model.generate(inputs.input_features.to("cpu"))

                msg = remove_repetitions(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip())

                if msg and msg not in ["", "you", ".", "..", "...", "Thank you."]:
                    speaker, callsign = characterize_speaker(msg)
                    msg_start = recording_time + timedelta(seconds=start_time)
                    msg_end = recording_time + timedelta(seconds=end_time)

                    transcription_queue.put({
                        "start_time": msg_start.isoformat(),
                        "end_time": msg_end.isoformat(),
                        "speaker": speaker,
                        "callsign": callsign,
                        "message": msg,
                        "audio_file": os.path.basename(filename),
                        "instructions": [m.lower() for m in ATC_VERB_REGEX.findall(msg)],
                        "metadata": {
                            "model": f"whisper-{Config.WHISPER_MODEL.split('/')[-1]}",
                            "processing_time": datetime.now().isoformat(),
                            "segment_start": start_time,
                            "segment_end": end_time,
                            "stream_title": audio_item.get("stream_title", "Unknown"),
                            "segment_index": seg_idx
                        }
                    })
                    logger.info(f"Chunk {chunk_num} seg {seg_idx}: '{msg[:50]}...'")

            cleanup_old_recordings()
            gc.collect()

        except Exception as e:
            logger.error(f"Error on chunk {chunk_num}: {e}")


def thread_output_writer():
    """Write transcriptions to Elasticsearch."""
    logger.info("Starting output writer...")

    while True:
        try:
            msg = transcription_queue.get(timeout=5)
        except queue.Empty:
            continue

        if msg is None:
            break

        logger.info(f"{'='*60}\nSpeaker: {msg['speaker']}\nMessage: {msg['message']}\n{'='*60}")
        index_to_elasticsearch(msg)


# Flask routes
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'elasticsearch_connected': es_client is not None and es_client.ping(),
        'model_loaded': model is not None
    })


@app.route('/status')
def status():
    return jsonify({
        'audio_queue_size': audio_queue.qsize(),
        'transcription_queue_size': transcription_queue.qsize(),
        'elasticsearch_connected': es_client is not None and es_client.ping(),
        'model_loaded': model is not None,
        'index': Config.ES_INDEX
    })


@app.route('/recordings')
def list_recordings():
    try:
        if not os.path.exists(Config.RECORDINGS_DIR):
            return jsonify({'recordings': []})

        recordings = [{
            'filename': f,
            'size_bytes': os.stat(os.path.join(Config.RECORDINGS_DIR, f)).st_size,
            'created_at': datetime.fromtimestamp(os.stat(os.path.join(Config.RECORDINGS_DIR, f)).st_birthtime).isoformat(),
            'url': f'/recordings/{f}'
        } for f in sorted(os.listdir(Config.RECORDINGS_DIR)) if f.endswith('.mp3')]

        return jsonify({'count': len(recordings), 'recordings': recordings})
    except Exception as e:
        logger.error(f"Error listing recordings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recordings/<filename>')
def get_recording(filename):
    try:
        filepath = os.path.join(Config.RECORDINGS_DIR, os.path.basename(filename))
        if not os.path.exists(filepath):
            abort(404, description='Recording not found')
        return send_file(filepath, mimetype='audio/mpeg', as_attachment=False, download_name=filename)
    except Exception as e:
        logger.error(f"Error serving recording: {e}")
        return jsonify({'error': str(e)}), 500


def start_transcription_threads():
    """Start all transcription threads."""
    logger.info("Starting transcription threads...")
    for target in [thread_stream_reader, thread_transcriber, thread_output_writer]:
        threading.Thread(target=target, daemon=True).start()
    logger.info("All transcription threads started")


if __name__ == '__main__':
    logger.info("Setting up Elasticsearch...")
    if setup_elasticsearch():
        logger.info("Elasticsearch setup complete")
    else:
        logger.warning("Elasticsearch setup failed - continuing without ES integration")

    start_transcription_threads()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
