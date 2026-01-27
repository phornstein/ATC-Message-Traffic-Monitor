#!/usr/bin/env python3
"""
Comprehensive setup script for ATC Voice to Text system
- Creates Elasticsearch indices and index templates
- Sets up ingest pipelines (ELSER)
- Deploys NER model using eland
"""
import os
import sys
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_es_client():
    """Initialize and return Elasticsearch client"""
    es_endpoint = os.getenv('ELASTICSEARCH_ENDPOINT')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')

    if not es_endpoint or not es_api_key:
        print("ERROR: ELASTICSEARCH_ENDPOINT or ELASTICSEARCH_API_KEY not set")
        sys.exit(1)

    es = Elasticsearch(
        es_endpoint,
        api_key=es_api_key,
        verify_certs=False
    )

    if not es.ping():
        print("ERROR: Could not connect to Elasticsearch")
        sys.exit(1)

    print(f"✓ Connected to Elasticsearch at {es_endpoint}")
    return es


def setup_composite_pipeline(es, deployed_models):
    """Create composite ingest pipeline with ELSER, NER, and Speaker Detection"""
    es_index = os.getenv('ELASTICSEARCH_INDEX', 'atc-transcriptions')
    elser_model_id = os.getenv('ELSER_MODEL_ID', 'elser_2')
    pipeline_name = f"{es_index}-composite-pipeline"

    # Build processors list - start with ELSER
    processors = [
        {
            "inference": {
                "model_id": elser_model_id,
                "input_output": [
                    {
                        "input_field": "message",
                        "output_field": "message_semantic"
                    }
                ]
            }
        }
    ]

    # Add NER model if deployed
    if "atc-ner-model" in deployed_models:
        processors.append({
            "inference": {
                "model_id": deployed_models["atc-ner-model"],
                "field_map": {
                    "message": "text_field"
                },
                "target_field": "ml.ner"
            }
        })

    # Add Speaker Detection model if deployed (text_classification)
    if "atc-speaker-model" in deployed_models:
        processors.append({
            "inference": {
                "model_id": deployed_models["atc-speaker-model"],
                "inference_config": {
                    "text_classification": {
                        "num_top_classes": 2
                    }
                },
                "field_map": {
                    "message": "text_field"
                },
                "target_field": "ml.speaker_role"
            }
        })

    pipeline_body = {
        "description": "Composite pipeline: ELSER + ATC NER + Speaker Detection",
        "processors": processors
    }

    try:
        es.ingest.get_pipeline(id=pipeline_name)
        print(f"✓ Pipeline '{pipeline_name}' already exists")
    except:
        es.ingest.put_pipeline(id=pipeline_name, body=pipeline_body)
        print(f"✓ Created composite ingest pipeline '{pipeline_name}'")
        print(f"  - ELSER semantic search")
        if "atc-ner-model" in deployed_models:
            print(f"  - ATC NER (entity extraction)")
        if "atc-speaker-model" in deployed_models:
            print(f"  - ATC Speaker Role Detection")

    return pipeline_name


def setup_transcription_index(es, pipeline_name):
    """Create transcription index with composite pipeline mapping"""
    es_index = os.getenv('ELASTICSEARCH_INDEX', 'atc-transcriptions')
    elser_model_id = os.getenv('ELSER_MODEL_ID', 'elser_2')

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
                    "type": "rank_features"
                },
                "audio_file": {"type": "keyword"},
                "instructions": {"type": "keyword"},
                "ml": {
                    "properties": {
                        "ner": {
                            "type": "object",
                            "enabled": True
                        },
                        "speaker_role": {
                            "type": "object",
                            "enabled": True
                        }
                    }
                },
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

    if es.indices.exists(index=es_index):
        print(f"✓ Index '{es_index}' already exists")
    else:
        es.indices.create(index=es_index, body=index_mapping)
        print(f"✓ Created index '{es_index}' with composite pipeline")


def setup_aircraft_template(es):
    """Create index template for aircraft data stream using logsdb"""
    aircraft_index = os.getenv('AIRCRAFT_INDEX', 'atc-aircraft')
    template_name = f"logs-{aircraft_index}-template"

    index_template = {
        "index_patterns": [f"logs-{aircraft_index}-*"],
        "data_stream": {},
        "priority": 500,
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "mode": "logsdb"
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "hex": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "flight": {"type": "keyword"},
                    "r": {"type": "keyword"},
                    "t": {"type": "keyword"},
                    "alt_baro": {"type": "keyword"},
                    "alt_geom": {"type": "keyword"},
                    "gs": {"type": "float"},
                    "track": {"type": "float"},
                    "baro_rate": {"type": "integer"},
                    "squawk": {"type": "keyword"},
                    "emergency": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "nav_qnh": {"type": "float"},
                    "nav_altitude_mcp": {"type": "integer"},
                    "nav_heading": {"type": "float"},
                    "lat": {"type": "float"},
                    "lon": {"type": "float"},
                    "location": {"type": "geo_point"},
                    "nic": {"type": "integer"},
                    "rc": {"type": "integer"},
                    "seen_pos": {"type": "float"},
                    "version": {"type": "integer"},
                    "nic_baro": {"type": "integer"},
                    "nac_p": {"type": "integer"},
                    "nac_v": {"type": "integer"},
                    "sil": {"type": "integer"},
                    "sil_type": {"type": "keyword"},
                    "gva": {"type": "integer"},
                    "sda": {"type": "integer"},
                    "alert": {"type": "integer"},
                    "spi": {"type": "integer"},
                    "messages": {"type": "long"},
                    "seen": {"type": "float"},
                    "rssi": {"type": "float"},
                    "dst": {"type": "float"},
                    "dir": {"type": "float"}
                }
            }
        }
    }

    try:
        es.indices.put_index_template(
            name=template_name,
            body=index_template
        )
        print(f"✓ Created/updated index template: {template_name}")
    except Exception as e:
        print(f"ERROR creating index template: {e}")
        sys.exit(1)


def deploy_ml_model(es, hf_model_id, es_model_id, task_type="ner"):
    """Deploy a single ML model using eland command-line tool"""
    # Check if model already exists
    try:
        model_info = es.ml.get_trained_models(model_id=es_model_id)
        print(f"✓ Model '{es_model_id}' already exists")
        return es_model_id
    except:
        print(f"Deploying model from HuggingFace: {hf_model_id}")

    # Check if eland is installed
    try:
        import subprocess
        result = subprocess.run(
            ["eland_import_hub_model", "--help"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: eland command-line tool not found. Installing eland...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "eland[pytorch]"])

    try:
        # Use eland command-line tool to import the model
        print("  - Downloading and uploading model to Elasticsearch...")
        print("    (This may take several minutes...)")

        import subprocess

        # Get Elasticsearch endpoint and API key
        es_endpoint = os.getenv('ELASTICSEARCH_ENDPOINT')
        es_api_key = os.getenv('ELASTICSEARCH_API_KEY')

        # Run eland_import_hub_model command
        cmd = [
            "eland_import_hub_model",
            "--url", es_endpoint,
            "--hub-model-id", hf_model_id,
            "--task-type", task_type,
            "--es-model-id", es_model_id,
            "--es-api-key", es_api_key,
            "--start"  # Start deployment automatically
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ Successfully deployed model '{es_model_id}'")
            print(f"  Model can be used in inference pipelines")
            return es_model_id
        else:
            print(f"ERROR deploying model:")
            print(result.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"ERROR: Model deployment timed out after 10 minutes")
        print(f"  The model may still be deploying in the background")
        return None
    except Exception as e:
        print(f"ERROR deploying model: {e}")
        print(f"\nNote: Model deployment requires:")
        print(f"  1. pip install 'eland[pytorch]'")
        print(f"  2. pip install transformers torch sentence-transformers")
        print(f"\nYou can deploy the model manually using:")
        print(f"  eland_import_hub_model \\")
        print(f"    --url {os.getenv('ELASTICSEARCH_ENDPOINT')} \\")
        print(f"    --hub-model-id {hf_model_id} \\")
        print(f"    --task-type {task_type} \\")
        print(f"    --es-model-id {es_model_id} \\")
        print(f"    --es-api-key YOUR_API_KEY \\")
        print(f"    --start")
        return None


def deploy_atc_models(es):
    """Deploy all ATC ML models (NER and Speaker Detection)"""
    print("\n=== Deploying ATC ML Models ===")

    models = [
        ("atc-ner-model", "Jzuluaga/bert-base-ner-atc-en-atco2-1h", "ner"),
        ("atc-speaker-model", "Jzuluaga/bert-base-speaker-role-atc-en-uwb-atcc", "text_classification")
    ]

    deployed_models = {}
    for es_model_id, hf_model_id, task_type in models:
        result = deploy_ml_model(es, hf_model_id, es_model_id, task_type=task_type)
        if result:
            deployed_models[es_model_id] = result

    return deployed_models




def main():
    """Main setup routine"""
    print("=" * 60)
    print("ATC Voice to Text - Elasticsearch Setup")
    print("=" * 60)

    # Connect to Elasticsearch
    es = get_es_client()

    # Deploy ATC ML models (NER and Speaker Detection)
    deployed_models = deploy_atc_models(es)

    # Setup composite pipeline with ELSER + NER + Speaker Detection
    print("\n=== Setting up Composite Ingest Pipeline ===")
    pipeline_name = setup_composite_pipeline(es, deployed_models)

    # Setup transcription index with composite pipeline
    print("\n=== Setting up Transcription Index ===")
    setup_transcription_index(es, pipeline_name)

    # Setup aircraft data stream
    print("\n=== Setting up Aircraft Data Stream ===")
    try:
        setup_aircraft_template(es)
        aircraft_index = os.getenv('AIRCRAFT_INDEX', 'atc-aircraft')
        print(f"  Data will be written to: logs-{aircraft_index}-default")
    except:
        print("ERROR setting up aircraft data stream")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the Docker containers: docker-compose up -d")
    print("2. Access the web UI at: http://localhost:3000")
    print("3. Monitor transcriptions in Elasticsearch")
    print("\nNote: This setup script uses requirements-setup.txt")
    print("The Docker containers use requirements.txt (without eland)")
    print("\nAll ATC transcriptions will automatically be enriched with:")
    print("  - ELSER semantic embeddings (for semantic search)")
    if "atc-ner-model" in deployed_models:
        print("  - Named Entity Recognition (callsigns, altitudes, headings, etc.)")
    if "atc-speaker-model" in deployed_models:
        print("  - Speaker Role Detection (pilot vs controller)")


if __name__ == "__main__":
    main()
