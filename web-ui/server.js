const express = require('express');
const { Client } = require('@elastic/elasticsearch');
const path = require('path');
const fetch = require('node-fetch');
require('dotenv').config({ path: '../.env' });

const app = express();
const port = process.env.WEB_UI_PORT || 3000;

// Elasticsearch client
const esClient = new Client({
    node: process.env.ELASTICSEARCH_ENDPOINT,
    auth: {
        apiKey: process.env.ELASTICSEARCH_API_KEY
    },
    tls: {
        rejectUnauthorized: true
    }
});

// Serve static files
app.use(express.static('public'));
app.use(express.json());

// API endpoint for aircraft data
app.get('/api/aircraft', async (req, res) => {
    try {
        const { start_time, end_time } = req.query;

        let timeQuery;
        if (start_time && end_time) {
            // Historical mode - use specific time range
            timeQuery = {
                range: {
                    '@timestamp': {
                        gte: start_time,
                        lte: end_time
                    }
                }
            };
        } else {
            // Live mode - last 30 seconds
            timeQuery = {
                range: {
                    '@timestamp': {
                        gte: 'now-30s'
                    }
                }
            };
        }

        const result = await esClient.search({
            index: process.env.AIRCRAFT_INDEX || 'atc-aircraft-korf',
            body: {
                query: timeQuery,
                size: 100,
                sort: [
                    { '@timestamp': 'desc' }
                ],
                collapse: {
                    field: 'hex'
                }
            }
        });

        const aircraft = result.hits.hits.map(hit => hit._source);
        res.json(aircraft);
    } catch (error) {
        console.error('Error fetching aircraft:', error);
        res.status(500).json({ error: 'Failed to fetch aircraft data' });
    }
});

// API endpoint for ATC messages
app.get('/api/messages', async (req, res) => {
    try {
        const { start_time, end_time } = req.query;

        let timeQuery;
        if (start_time && end_time) {
            // Historical mode - use specific time range
            timeQuery = {
                range: {
                    'start_time': {
                        gte: start_time,
                        lte: end_time
                    }
                }
            };
        } else {
            // Live mode - last 10 minutes
            timeQuery = {
                range: {
                    '@timestamp': {
                        gte: 'now-10m'
                    }
                }
            };
        }

        const result = await esClient.search({
            index: process.env.ELASTICSEARCH_INDEX || 'atc-transcription-korf',
            body: {
                query: timeQuery,
                size: 50,
                sort: [
                    { 'start_time': 'desc' }
                ]
            }
        });

        const messages = result.hits.hits.map(hit => hit._source);
        res.json(messages);
    } catch (error) {
        console.error('Error fetching messages:', error);
        res.status(500).json({ error: 'Failed to fetch messages' });
    }
});

// API endpoint for semantic search
app.post('/api/messages/search', async (req, res) => {
    try {
        const { query, start_time, end_time } = req.body;

        if (!query || query.trim() === '') {
            return res.status(400).json({ error: 'Search query is required' });
        }

        const elserModelId = process.env.ELSER_MODEL_ID || 'elser_2';

        // Build the query
        const mustQueries = [
            {
                text_expansion: {
                    'message_semantic': {
                        model_id: elserModelId,
                        model_text: query
                    }
                }
            }
        ];

        // Add time range if provided
        if (start_time && end_time) {
            mustQueries.push({
                range: {
                    'start_time': {
                        gte: start_time,
                        lte: end_time
                    }
                }
            });
        }

        const result = await esClient.search({
            index: process.env.ELASTICSEARCH_INDEX || 'atc-transcription-korf',
            body: {
                query: {
                    bool: {
                        must: mustQueries
                    }
                },
                size: 50,
                sort: [
                    { '_score': 'desc' },
                    { 'start_time': 'desc' }
                ]
            }
        });

        const messages = result.hits.hits.map(hit => hit._source);
        res.json(messages);
    } catch (error) {
        console.error('Error performing semantic search:', error);
        res.status(500).json({ error: 'Failed to perform semantic search' });
    }
});

// Audio proxy endpoint
app.get('/api/audio/:filename', async (req, res) => {
    try {
        const { filename } = req.params;
        const atcServerUrl = process.env.ATC_SERVER_URL || 'http://atc-transcription:8000';
        const audioUrl = `${atcServerUrl}/recordings/${filename}`;

        const response = await fetch(audioUrl);

        if (!response.ok) {
            return res.status(response.status).json({
                error: 'Failed to fetch audio file',
                details: response.statusText
            });
        }

        // Stream the audio file
        res.setHeader('Content-Type', response.headers.get('content-type') || 'audio/mpeg');
        res.setHeader('Content-Length', response.headers.get('content-length'));
        res.setHeader('Accept-Ranges', 'bytes');

        response.body.pipe(res);
    } catch (error) {
        console.error('Error proxying audio:', error);
        res.status(500).json({ error: 'Failed to proxy audio file' });
    }
});

// ATC tower configuration endpoint
app.get('/api/config', (req, res) => {
    res.json({
        atc_lat: parseFloat(process.env.ATC_LAT) || 36.89,
        atc_lon: parseFloat(process.env.ATC_LON) || -76.2
    });
});

// Health check
app.get('/health', async (req, res) => {
    try {
        const health = await esClient.ping();
        res.json({
            status: 'healthy',
            elasticsearch: health ? 'connected' : 'disconnected'
        });
    } catch (error) {
        res.status(500).json({
            status: 'unhealthy',
            elasticsearch: 'disconnected',
            error: error.message
        });
    }
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`ATC Web UI running on http://0.0.0.0:${port}`);
    console.log(`Elasticsearch: ${process.env.ELASTICSEARCH_ENDPOINT}`);
    console.log(`Aircraft Index: ${process.env.AIRCRAFT_INDEX}`);
    console.log(`Messages Index: ${process.env.ELASTICSEARCH_INDEX}`);
});
