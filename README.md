# ATC Voice to Text - Setup Guide

This guide walks you through setting up the ATC Voice to Text system with Elasticsearch integration, aircraft tracking, and NER model deployment.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Elasticsearch cluster with API access
- API key for Elasticsearch

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `ELASTICSEARCH_ENDPOINT` - Your Elasticsearch cluster URL
- `ELASTICSEARCH_API_KEY` - Your Elasticsearch API key
- `ELASTICSEARCH_INDEX` - Index name for transcriptions (default: atc-transcriptions)
- `AIRCRAFT_INDEX` - Base name for aircraft data stream (default: atc-aircraft). The full data stream name will be `logs-{AIRCRAFT_INDEX}-default` using logsdb mode.
- `ATC_LAT`, `ATC_LON`, `ATC_RADIUS_NM` - Location and radius for aircraft tracking
- `STREAM_URL` - URL of the ATC audio stream
- `ELSER_MODEL_ID` - ELSER model ID (default: elser_2)

### 2. Run Setup Script

The setup script will:
- Create Elasticsearch indices and mappings
- Set up ELSER pipeline for semantic search
- Deploy the ATC NER model from HuggingFace
- Create NER pipeline for entity extraction

```bash
# Install setup dependencies
pip install -r requirements-setup.txt

# Run setup
python setup.py
```

**Note:** The setup script uses `requirements-setup.txt` which includes eland for model deployment. The Docker containers use `transcription-server/requirements.txt` which doesn't include eland to avoid dependency conflicts with Whisper.

### 3. Start Docker Containers

```bash
docker-compose up -d
```

This will start:
- **atc-transcription**: Whisper transcription service (from `transcription-server/`)
- **atc-aircraft**: Logstash pipeline for aircraft tracking (from `logstash/`)
- **web-ui**: Web interface for viewing data (from `web-ui/`)

**Note:** The Whisper model is cached in a Docker volume (`huggingface-cache`). On first run, it will download the model (~3GB). Subsequent container restarts will use the cached model.

### 4. Access the Web UI

Open your browser to: http://localhost:3000

## Features

### Transcription Index
- ELSER semantic search on ATC messages
- Audio playback of original recordings
- Speaker and callsign detection
- Instruction keyword extraction

### Aircraft Tracking
- Real-time aircraft positions from ADS-B data
- Uses Elasticsearch data streams with logsdb mode for optimized time-series storage
- Data stream naming: `logs-{AIRCRAFT_INDEX}-default`
- Geo-point mapping for location queries
- Historical track playback

### NER Model
- ATC-specific Named Entity Recognition
- Extracts callsigns, locations, altitudes, headings
- Model: `Jzuluaga/bert-base-ner-atc-en-atco2-1h`

## Manual NER Model Deployment

If the automatic deployment fails, you can deploy manually:

```bash
eland_import_hub_model \
  --url YOUR_ELASTICSEARCH_ENDPOINT \
  --hub-model-id Jzuluaga/bert-base-ner-atc-en-atco2-1h \
  --task-type ner \
  --es-model-id atc-ner-model \
  --es-api-key YOUR_API_KEY \
  --start
```

## Troubleshooting

### Docker Build Fails
Make sure you're using the correct requirements file. The transcription server uses `transcription-server/requirements.txt` (without eland).

### Setup Script Fails
- Ensure you have Python 3.11+
- Install setup dependencies: `pip install -r requirements-setup.txt`
- Check your Elasticsearch credentials in `.env`

### NER Model Deployment Times Out
The model download can take 10+ minutes. You can:
1. Wait for it to complete in the background
2. Deploy manually using the `eland_import_hub_model` command above
3. Skip NER deployment and use only ELSER for semantic search

## Project Structure

```
ATC Voice to Text/
├── transcription-server/       # Python transcription service
│   ├── app.py                  # Main server application
│   ├── Dockerfile
│   ├── requirements.txt
│   └── recordings/             # Audio chunk storage
├── logstash/                   # Aircraft tracking
│   └── pipeline/
│       └── aircraft.conf
├── web-ui/                     # Node.js frontend
│   ├── server.js
│   ├── Dockerfile
│   ├── package.json
│   └── public/
├── setup.py                    # Elasticsearch setup
├── requirements-setup.txt      # Setup dependencies
├── docker-compose.yml
├── .env
└── .env.example
```

## Architecture

```
┌─────────────────┐
│  ATC Stream     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Whisper AI     │◄── Transcription Service
│  (Docker)       │    (transcription-server/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Elasticsearch  │◄── ELSER + NER Pipelines
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web UI         │◄── Live/Replay Mode
│  (Leaflet Map)  │    Semantic Search
└─────────────────┘    Audio Playback
```

## Next Steps

1. Monitor the transcription service: `docker logs -f atc`
2. Check aircraft tracking: `docker logs -f aircraft`
3. View the web UI at http://localhost:3000
4. Test semantic search in the UI
5. Try replay mode to view historical data
