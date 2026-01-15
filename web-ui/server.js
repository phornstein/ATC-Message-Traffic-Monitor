const express = require('express');
const { Client } = require('@elastic/elasticsearch');
const fetch = require('node-fetch');
require('dotenv').config({ path: '../.env' });

const app = express();
const port = process.env.WEB_UI_PORT || 3000;

// Elasticsearch client
const esClient = new Client({
    node: process.env.ELASTICSEARCH_ENDPOINT,
    auth: { apiKey: process.env.ELASTICSEARCH_API_KEY },
    tls: { rejectUnauthorized: true }
});

// Index names
const AIRCRAFT_INDEX = `logs-${process.env.AIRCRAFT_INDEX || 'atc-aircraft-korf'}-default`;
const MESSAGES_INDEX = process.env.ELASTICSEARCH_INDEX || 'atc-transcription-korf';

app.use(express.static('public'));
app.use(express.json());

// Helper: Build time range query
function buildTimeQuery(startTime, endTime, field = '@timestamp', liveRange = 'now-30s') {
    if (startTime && endTime) {
        return { range: { [field]: { gte: startTime, lte: endTime } } };
    }
    return { range: { [field]: { gte: liveRange } } };
}

// Helper: Execute ES search and return sources
async function searchES(index, body) {
    const result = await esClient.search({ index, body });
    return result.hits.hits.map(hit => hit._source);
}

// API: Aircraft data
app.get('/api/aircraft', async (req, res) => {
    try {
        const { start_time, end_time } = req.query;
        const data = await searchES(AIRCRAFT_INDEX, {
            query: buildTimeQuery(start_time, end_time),
            size: 100,
            sort: [{ '@timestamp': 'desc' }],
            collapse: { field: 'hex' }
        });
        res.json(data);
    } catch (error) {
        console.error('Error fetching aircraft:', error);
        res.status(500).json({ error: 'Failed to fetch aircraft data' });
    }
});

// API: ATC messages
app.get('/api/messages', async (req, res) => {
    try {
        const { start_time, end_time } = req.query;
        const data = await searchES(MESSAGES_INDEX, {
            query: buildTimeQuery(start_time, end_time, '@timestamp', 'now-10m'),
            size: 50,
            sort: [{ '@timestamp': 'desc' }]
        });
        res.json(data);
    } catch (error) {
        console.error('Error fetching messages:', error);
        res.status(500).json({ error: 'Failed to fetch messages' });
    }
});

// API: Semantic search
app.post('/api/messages/search', async (req, res) => {
    try {
        const { query, start_time, end_time } = req.body;
        if (!query?.trim()) {
            return res.status(400).json({ error: 'Search query is required' });
        }

        const mustQueries = [{
            text_expansion: {
                'message_semantic': {
                    model_id: process.env.ELSER_MODEL_ID || 'elser_2',
                    model_text: query
                }
            }
        }];

        if (start_time && end_time) {
            mustQueries.push({ range: { '@timestamp': { gte: start_time, lte: end_time } } });
        }

        const data = await searchES(MESSAGES_INDEX, {
            query: { bool: { must: mustQueries } },
            size: 50,
            sort: [{ '_score': 'desc' }, { '@timestamp': 'desc' }]
        });
        res.json(data);
    } catch (error) {
        console.error('Error performing semantic search:', error);
        res.status(500).json({ error: 'Failed to perform semantic search' });
    }
});

// API: Audio proxy
app.get('/api/audio/:filename', async (req, res) => {
    try {
        const atcServerUrl = process.env.ATC_SERVER_URL || 'http://atc-transcription:8000';
        const response = await fetch(`${atcServerUrl}/recordings/${req.params.filename}`);

        if (!response.ok) {
            return res.status(response.status).json({ error: 'Failed to fetch audio file' });
        }

        res.setHeader('Content-Type', response.headers.get('content-type') || 'audio/mpeg');
        res.setHeader('Content-Length', response.headers.get('content-length'));
        res.setHeader('Accept-Ranges', 'bytes');
        response.body.pipe(res);
    } catch (error) {
        console.error('Error proxying audio:', error);
        res.status(500).json({ error: 'Failed to proxy audio file' });
    }
});

// API: Configuration
app.get('/api/config', (req, res) => {
    res.json({
        atc_lat: parseFloat(process.env.ATC_LAT) || 36.89,
        atc_lon: parseFloat(process.env.ATC_LON) || -76.2
    });
});

// API: Timeline density for gap visualization
app.get('/api/timeline/density', async (req, res) => {
    try {
        const { start_time, end_time, bucket_size = 5 } = req.query;

        if (!start_time || !end_time) {
            return res.status(400).json({ error: 'start_time and end_time are required' });
        }

        const result = await esClient.search({
            index: AIRCRAFT_INDEX,
            body: {
                size: 0,
                query: { range: { '@timestamp': { gte: start_time, lte: end_time } } },
                aggs: {
                    timeline: {
                        date_histogram: {
                            field: '@timestamp',
                            fixed_interval: `${bucket_size}m`,
                            min_doc_count: 0,
                            extended_bounds: { min: start_time, max: end_time }
                        }
                    }
                }
            }
        });

        const buckets = result.aggregations.timeline.buckets.map(b => ({
            time: b.key_as_string,
            timestamp: b.key,
            count: b.doc_count,
            hasData: b.doc_count > 0
        }));

        res.json({ buckets, totalBuckets: buckets.length });
    } catch (error) {
        console.error('Error fetching timeline density:', error);
        res.status(500).json({ error: 'Failed to fetch timeline density' });
    }
});

// Health check
app.get('/health', async (req, res) => {
    try {
        const health = await esClient.ping();
        res.json({ status: 'healthy', elasticsearch: health ? 'connected' : 'disconnected' });
    } catch (error) {
        res.status(500).json({ status: 'unhealthy', elasticsearch: 'disconnected', error: error.message });
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`ATC Web UI running on http://0.0.0.0:${port}`);
    console.log(`Elasticsearch: ${process.env.ELASTICSEARCH_ENDPOINT}`);
    console.log(`Aircraft Index: ${AIRCRAFT_INDEX}`);
    console.log(`Messages Index: ${MESSAGES_INDEX}`);
});
