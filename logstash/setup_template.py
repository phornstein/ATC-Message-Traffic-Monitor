#!/usr/bin/env python3
"""
Setup Elasticsearch index template for aircraft data stream
"""
import os
import sys
from elasticsearch import Elasticsearch

def setup_aircraft_template():
    """Create index template for aircraft data stream"""

    # Get configuration from environment
    es_endpoint = os.getenv('ELASTICSEARCH_ENDPOINT')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')
    aircraft_index = os.getenv('AIRCRAFT_INDEX', 'atc-aircraft')

    if not es_endpoint or not es_api_key:
        print("ERROR: ELASTICSEARCH_ENDPOINT or ELASTICSEARCH_API_KEY not set")
        sys.exit(1)

    # Initialize Elasticsearch client
    es = Elasticsearch(
        es_endpoint,
        api_key=es_api_key
    )

    # Test connection
    if not es.ping():
        print("ERROR: Could not connect to Elasticsearch")
        sys.exit(1)

    print(f"Connected to Elasticsearch at {es_endpoint}")

    # Index template for data stream
    template_name = f"{aircraft_index}-template"

    index_template = {
        "index_patterns": [f"atc-aircraft-*"],
        "data_stream": {},
        "priority": 500,
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "hex": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "flight": {"type": "keyword"},
                    "r": {"type": "keyword"},
                    "t": {"type": "keyword"},
                    "alt_baro": {"type": "integer"},
                    "alt_geom": {"type": "integer"},
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

    # Create or update template
    try:
        es.indices.put_index_template(
            name=template_name,
            body=index_template
        )
        print(f"✓ Created/updated index template: {template_name}")
    except Exception as e:
        print(f"ERROR creating index template: {e}")
        sys.exit(1)

    print(f"✓ Aircraft data stream setup complete")
    print(f"  Data will be written to: {aircraft_index}")

if __name__ == "__main__":
    setup_aircraft_template()
