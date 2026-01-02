from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import os
import threading
import queue
import requests
import re
import json
import gc
import torch
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from flask import Flask, jsonify, send_file, abort
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import logging

# Load environment variables
load_dotenv()

# Set environment variables for PyTorch to limit memory usage
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global configuration
RECORDINGS_DIR = os.getenv('RECORDINGS_DIR', 'recordings')
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

CHUNK_DURATION = int(os.getenv('CHUNK_DURATION', 30))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 8192))
MAX_BUFFER_SIZE = 50
MAX_RECORDINGS = int(os.getenv('MAX_RECORDINGS', 1000))  # Keep only last N recordings

# Elasticsearch configuration
ES_ENDPOINT = os.getenv('ELASTICSEARCH_ENDPOINT')
ES_API_KEY = os.getenv('ELASTICSEARCH_API_KEY')
ES_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'atc-transcriptions')
ELSER_MODEL_ID = os.getenv('ELSER_MODEL_ID', 'elser_2')

# Whisper ATC model configuration
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'jlvdoorn/whisper-large-v3-atco2-asr')
OVERLAP_DURATION = int(os.getenv('OVERLAP_DURATION', 5))  # seconds of overlap between chunks

# VAD (Voice Activity Detection) configuration
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', '0.02'))  # Energy threshold for voice detection
MIN_SILENCE_DURATION = float(os.getenv('MIN_SILENCE_DURATION', '0.5'))  # Minimum silence to split (seconds)
MIN_SPEECH_DURATION = float(os.getenv('MIN_SPEECH_DURATION', '0.3'))  # Minimum speech segment duration

# Global queues and model
audio_queue = queue.Queue(maxsize=100)
transcription_queue = queue.Queue(maxsize=50)
model = None
processor = None
es_client = None

# ATC callsign regex (case-insensitive)
CALLSIGN_REGEX = re.compile(
    r"\b(?:"
    r"(?!CONTACT\b|CLEAR\b|CLEARED\b|CLIMB\b|DESCEND\b|MAINTAIN\b|TURN\b|PROCEED\b|HOLD\b|TAXI\b|LINE\b|WAIT\b|EXPECT\b)"
    r"[A-Z]{3,15}\s+\d{2,4}"
    r"|"
    r"[A-Z]{2,3}\d{2,4}"
    r"|"
    r"(?:N|C|G|D|F|I|EC|JA|VH|ZS|OE|HB)(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}"
    r")\b",
    re.IGNORECASE
)

ATC_VERBS = {
    "contact", "clear", "cleared", "climb", "descend",
    "maintain", "turn", "proceed", "hold", "taxi",
    "line", "wait", "expect"
}

ATC_VERB_REGEX = re.compile(
    r"\b(" + "|".join(sorted(ATC_VERBS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)


def detect_speech_segments(audio, sr):
    """
    Detect speech segments in audio using energy-based VAD.
    Returns list of (start_time, end_time) tuples in seconds.
    """
    # Calculate frame-level energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    # Calculate RMS energy for each frame
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize energy
    energy = energy / (np.max(energy) + 1e-8)

    # Detect voice activity
    is_speech = energy > VAD_THRESHOLD

    # Convert to time segments
    segments = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            # Start of speech
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            # End of speech
            start_time = start_frame * hop_length / sr
            end_time = i * hop_length / sr

            # Only keep segments longer than minimum duration
            if (end_time - start_time) >= MIN_SPEECH_DURATION:
                segments.append((start_time, end_time))

            in_speech = False

    # Handle case where speech continues to end
    if in_speech:
        start_time = start_frame * hop_length / sr
        end_time = len(audio) / sr
        if (end_time - start_time) >= MIN_SPEECH_DURATION:
            segments.append((start_time, end_time))

    # Merge segments that are too close together
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start - current_end < MIN_SILENCE_DURATION:
                # Merge with current segment
                current_end = end
            else:
                # Save current segment and start new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end

        # Add final segment
        merged_segments.append((current_start, current_end))

    return merged_segments


def extract_instruction_keywords(message: str):
    """
    Returns a list of ATC instruction keywords found in the message,
    or an empty list if none are found.
    """
    if not message:
        return []

    matches = ATC_VERB_REGEX.findall(message)
    return [m.lower() for m in matches]


def extract_callsign(text):
    match = CALLSIGN_REGEX.search(text)
    if not match:
        return None

    cs = match.group(0)

    # Reject pure words
    if cs.isalpha():
        return None

    return cs


def characterize_speaker(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]

    # 1. Check first clause (aircraft calling ATC)
    if parts:
        cs = extract_callsign(parts[0])
        if cs and cs.lower() not in ATC_VERBS:
            return "Aircraft", cs

    # 2. Check last clause (ATC addressing aircraft)
    if parts:
        cs = extract_callsign(parts[-1])
        if cs and cs.lower() not in ATC_VERBS:
            return "Tower", cs

    return "Unknown", None


def cleanup_old_recordings():
    """Remove old recordings to save disk space and memory"""
    try:
        if not os.path.exists(RECORDINGS_DIR):
            return

        recordings = []
        for filename in os.listdir(RECORDINGS_DIR):
            if filename.endswith('.mp3'):
                filepath = os.path.join(RECORDINGS_DIR, filename)
                recordings.append((filepath, os.path.getmtime(filepath)))

        # Sort by modification time (oldest first)
        recordings.sort(key=lambda x: x[1])

        # Remove old recordings if we exceed the limit
        if len(recordings) > MAX_RECORDINGS:
            to_remove = recordings[:len(recordings) - MAX_RECORDINGS]
            for filepath, _ in to_remove:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old recording: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.error(f"Failed to remove {filepath}: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up recordings: {e}")


def setup_elasticsearch():
    """Initialize Elasticsearch client and setup index with ELSER pipeline"""
    global es_client

    if not ES_ENDPOINT or not ES_API_KEY:
        logger.error("Elasticsearch endpoint or API key not configured")
        return False

    try:
        # Initialize Elasticsearch client
        es_client = Elasticsearch(
            ES_ENDPOINT,
            api_key=ES_API_KEY
        )

        # Test connection
        if not es_client.ping():
            logger.error("Could not connect to Elasticsearch")
            return False

        logger.info("Successfully connected to Elasticsearch")

        # Create ingest pipeline with ELSER
        pipeline_name = f"{ES_INDEX}-elser-pipeline"
        pipeline_body = {
            "description": "ELSER semantic text pipeline for ATC transcriptions",
            "processors": [
                {
                    "inference": {
                        "model_id": ELSER_MODEL_ID,
                        "input_output": [
                            {
                                "input_field": "message",
                                "output_field": "message_semantic"
                            }
                        ]
                    }
                }
            ]
        }

        # Check if pipeline exists
        try:
            es_client.ingest.get_pipeline(id=pipeline_name)
            logger.info(f"Pipeline '{pipeline_name}' already exists")
        except:
            # Create pipeline
            es_client.ingest.put_pipeline(id=pipeline_name, body=pipeline_body)
            logger.info(f"Created ingest pipeline '{pipeline_name}'")

        # Create index with mapping
        index_mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "default_pipeline": pipeline_name
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "start_time": {"type": "date"},
                    "end_time": {"type": "date"},
                    "speaker": {"type": "keyword"},
                    "callsign": {"type": "keyword"},
                    "message": {"type": "text"},
                    "message_semantic": {
                        "type": "rank_features",
                        "inference_id": ELSER_MODEL_ID
                    },
                    "audio_file": {"type": "keyword"},
                    "instructions": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "model": {"type": "keyword"},
                            "processing_time": {"type": "date"},
                            "segment_start": {"type": "float"},
                            "segment_end": {"type": "float"},
                            "stream_title": {"type": "text"}
                        }
                    }
                }
            }
        }

        # Check if index exists
        if not es_client.indices.exists(index=ES_INDEX):
            es_client.indices.create(index=ES_INDEX, body=index_mapping)
            logger.info(f"Created index '{ES_INDEX}' with ELSER pipeline")
        else:
            logger.info(f"Index '{ES_INDEX}' already exists")

        return True

    except Exception as e:
        logger.error(f"Error setting up Elasticsearch: {e}")
        return False


def index_to_elasticsearch(data):
    """Index transcription data to Elasticsearch"""
    global es_client

    if not es_client:
        logger.warning("Elasticsearch client not initialized, skipping indexing")
        return

    try:
        # Add timestamp
        data['@timestamp'] = datetime.now().isoformat()

        # Index document
        response = es_client.index(index=ES_INDEX, document=data)
        logger.info(f"Indexed document to Elasticsearch: {response['_id']}")

    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {e}")


def thread_stream_reader():
    """Thread 1: Read from live ATC stream"""
    playlist_url = os.getenv('ATC_PLAYLIST_URL', 'https://www.liveatc.net/play/korf1.pls')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'https://www.liveatc.net/'
    }

    logger.info("Starting stream reader...")

    try:
        # Fetch playlist
        playlist_response = requests.get(playlist_url, timeout=10, headers=headers)
        playlist_content = playlist_response.text
        stream_urls = re.findall(r'File\d+=(.+?)(?:\n|$)', playlist_content)
        stream_title = re.findall(r'Title\d+=(.+?)(?:\n|$)', playlist_content)

        if not stream_urls:
            logger.error("Could not find stream URL")
            return

        audio_url = stream_urls[0].strip()
        logger.info(f"Found stream: {audio_url}")

        # Connect to stream
        response = requests.get(audio_url, stream=True, timeout=None, headers=headers)
        response.raise_for_status()

        # Get bitrate from headers
        bitrate_str = response.headers.get('icy-br', '32')
        bitrate_kbps = int(bitrate_str)
        bytes_per_second = (bitrate_kbps * 1000) / 8
        chunk_size_bytes = int(CHUNK_DURATION * bytes_per_second)

        logger.info(f"Bitrate: {bitrate_kbps} kbps | Chunk size: {chunk_size_bytes} bytes")

        # Calculate overlap in bytes
        overlap_size_bytes = int(OVERLAP_DURATION * bytes_per_second)
        logger.info(f"Overlap duration: {OVERLAP_DURATION}s | Overlap size: {overlap_size_bytes} bytes")

        chunk_number = 0
        buffer = b""
        overlap_buffer = b""

        for data in response.iter_content(chunk_size=CHUNK_SIZE):
            if data:
                buffer += data

                if len(buffer) >= chunk_size_bytes:
                    chunk_number += 1
                    timestamp = datetime.now().timestamp()
                    # Generate timestamp-based filename
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    chunk_filename = os.path.join(RECORDINGS_DIR, f"atc_{timestamp_str}.mp3")

                    # Create chunk with overlap from previous chunk
                    chunk_data = overlap_buffer + buffer[:chunk_size_bytes]

                    # Write chunk to file (with overlap)
                    with open(chunk_filename, 'wb') as f:
                        f.write(chunk_data)

                    # Send to queue
                    audio_queue.put({
                        'chunk_num': chunk_number,
                        'filename': chunk_filename,
                        'timestamp': timestamp,
                        "stream_title": stream_title[0] if stream_title else "Unknown",
                        "has_overlap": len(overlap_buffer) > 0
                    })

                    logger.info(f"Queued chunk {chunk_number} at {timestamp} (overlap: {len(overlap_buffer)} bytes)")

                    # Store the last OVERLAP_DURATION seconds for next chunk
                    if len(buffer) >= overlap_size_bytes:
                        overlap_buffer = buffer[chunk_size_bytes - overlap_size_bytes:chunk_size_bytes]
                    else:
                        overlap_buffer = buffer[:chunk_size_bytes]

                    buffer = buffer[chunk_size_bytes:]

    except KeyboardInterrupt:
        logger.info("Stream interrupted")
    except Exception as e:
        logger.error(f"Stream error: {type(e).__name__}: {e}")


def thread_transcriber():
    """Thread 2: Transcribe audio chunks"""
    global model, processor

    logger.info("Starting transcriber...")
    logger.info(f"Loading Whisper ATC model: {WHISPER_MODEL}")

    try:
        # Disable gradient computation globally to save memory
        torch.set_grad_enabled(False)

        # Set memory allocation strategy
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)

        device = "cpu"
        logger.info(f"Using device: {device}")

        # Load Whisper model and processor
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL)

        # Move model to CPU explicitly and set to eval mode
        model = model.to(device)
        model.eval()

        logger.info("Whisper ATC model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
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
                # Load audio file using librosa (handles MP3 natively)
                audio_array, file_sampling_rate = librosa.load(filename, sr=16000, mono=True)

                # Detect speech segments using VAD
                segments = detect_speech_segments(audio_array, 16000)

                if not segments:
                    logger.info(f"Chunk {chunk_num}: No speech detected")
                    continue

                logger.info(f"Chunk {chunk_num}: Detected {len(segments)} speech segment(s)")

                # Process each speech segment
                for seg_idx, (start_time, end_time) in enumerate(segments):
                    # Extract audio segment
                    start_sample = int(start_time * 16000)
                    end_sample = int(end_time * 16000)
                    segment_audio = audio_array[start_sample:end_sample]

                    # Skip very short segments
                    if len(segment_audio) < 16000 * 0.3:  # Less than 0.3 seconds
                        continue

                    # Process audio with the Whisper processor
                    inputs = processor(segment_audio, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs.input_features.to(device)

                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = model.generate(input_features)

                    # Decode transcription
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    msg = transcription.strip()

                    # Skip empty or trivial transcriptions
                    if msg and msg not in ["", "you", ".", "..", "...", "Thank you."]:
                        speaker, callsign = characterize_speaker(msg)

                        # Calculate absolute timestamps
                        msg_start = recording_time + timedelta(seconds=start_time)
                        msg_end = recording_time + timedelta(seconds=end_time)

                        data = {
                            "start_time": msg_start.isoformat(),
                            "end_time": msg_end.isoformat(),
                            "speaker": speaker,
                            "callsign": callsign,
                            "message": msg,
                            "audio_file": os.path.basename(filename),
                            "instructions": extract_instruction_keywords(msg),
                            "metadata": {
                                "model": f"whisper-{WHISPER_MODEL.split('/')[-1]}",
                                "processing_time": datetime.now().isoformat(),
                                "segment_start": start_time,
                                "segment_end": end_time,
                                "stream_title": audio_item.get("stream_title", "Unknown"),
                                "segment_index": seg_idx
                            }
                        }

                        transcription_queue.put(data)
                        logger.info(f"Chunk {chunk_num} segment {seg_idx}: '{msg[:50]}...'")

                # Cleanup old recordings and force garbage collection
                cleanup_old_recordings()
                gc.collect()

            except Exception as e:
                logger.error(f"Error on chunk {chunk_num}: {e}")

    except Exception as e:
        logger.error(f"Fatal transcriber error: {e}")


def thread_output_writer():
    """Thread 3: Write transcriptions to Elasticsearch"""
    logger.info("Starting output writer...")

    try:
        while True:
            try:
                msg = transcription_queue.get(timeout=5)
            except queue.Empty:
                continue

            if msg is None:
                break

            # Display
            logger.info("="*80)
            logger.info(f"Speaker: {msg['speaker']}")
            logger.info(f"Message: {msg['message']}")
            logger.info("="*80)

            # Index to Elasticsearch
            index_to_elasticsearch(msg)

    except Exception as e:
        logger.error(f"Output writer error: {e}")


# Flask routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'elasticsearch_connected': es_client is not None and es_client.ping(),
        'model_loaded': model is not None
    })


@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        'audio_queue_size': audio_queue.qsize(),
        'transcription_queue_size': transcription_queue.qsize(),
        'elasticsearch_connected': es_client is not None and es_client.ping(),
        'model_loaded': model is not None,
        'index': ES_INDEX
    })


@app.route('/recordings', methods=['GET'])
def list_recordings():
    """List all available audio chunk recordings"""
    try:
        if not os.path.exists(RECORDINGS_DIR):
            return jsonify({'recordings': []})

        recordings = []
        for filename in sorted(os.listdir(RECORDINGS_DIR)):
            if filename.endswith('.mp3'):
                filepath = os.path.join(RECORDINGS_DIR, filename)
                stat_info = os.stat(filepath)
                recordings.append({
                    'filename': filename,
                    'size_bytes': stat_info.st_size,
                    'created_at': datetime.fromtimestamp(stat_info.st_birthtime).isoformat(),
                    'url': f'/recordings/{filename}'
                })

        return jsonify({
            'count': len(recordings),
            'recordings': recordings
        })

    except Exception as e:
        logger.error(f"Error listing recordings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recordings/<filename>', methods=['GET'])
def get_recording(filename):
    """Download a specific audio chunk recording"""
    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)

        filepath = os.path.join(RECORDINGS_DIR, filename)

        if not os.path.exists(filepath):
            abort(404, description='Recording not found')

        return send_file(
            filepath,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error serving recording: {e}")
        return jsonify({'error': str(e)}), 500


def start_transcription_threads():
    """Start all transcription threads"""
    logger.info("Starting transcription threads...")

    t1 = threading.Thread(target=thread_stream_reader, daemon=True)
    t2 = threading.Thread(target=thread_transcriber, daemon=True)
    t3 = threading.Thread(target=thread_output_writer, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    logger.info("All transcription threads started")


if __name__ == '__main__':
    # Setup Elasticsearch
    logger.info("Setting up Elasticsearch...")
    if setup_elasticsearch():
        logger.info("Elasticsearch setup complete")
    else:
        logger.warning("Elasticsearch setup failed - continuing without ES integration")

    # Start transcription threads
    start_transcription_threads()

    # Start Flask server
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))

    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
