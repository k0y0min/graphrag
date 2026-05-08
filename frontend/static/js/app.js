// Backend configuration
const BACKEND_URL = window.BACKEND_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');
const POLL_INTERVAL = 2000; // 2 seconds

// Graph globals
let network = null;
let edgesDS = null;
let nodesDS = null;
let nodesView = null;
let edgesView = null;

// DOM Elements
const ingestBtn = document.getElementById('ingest-btn');
const ingestText = document.getElementById('ingest-text');
const clearDbBtn = document.getElementById('clear-db-btn');
const ingestStatus = document.getElementById('ingest-status');
const progressContainer = document.getElementById('progress-container');


const queryBtn = document.getElementById('query-btn');
const queryInput = document.getElementById('query-input');
const queryResults = document.getElementById('query-results');


const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Init
document.addEventListener('DOMContentLoaded', () => {
    checkBackendReady();
});

async function checkBackendReady() {
    const overlay = document.getElementById('loading-overlay');

    while (true) {
        try {
            const res = await fetch(`${BACKEND_URL}/health`);
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'healthy') {
                    console.log("Backend is healthy!");
                    overlay.classList.add('hidden');
                    initGraph();
                    break;
                }
            }
        } catch (e) {
            console.log("Waiting for backend...");
        }
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
    }
}

// Tabs
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.add('hidden'));

        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`${tabId}-tab`).classList.remove('hidden');
    });
});

// Graph Initialization
function initGraph() {
    const container = document.getElementById('network-container');

    // Sync initial values from DOM
    const initialSpringLength = parseInt(document.getElementById('spring-length-range')?.value || 100);
    const initialGravity = parseInt(document.getElementById('gravity-range')?.value || -2000);
    const initialPhysics = document.getElementById('physics-toggle')?.checked ?? true;
    const initialEntityLabels = document.getElementById('entity-labels-toggle')?.checked ?? true;
    const initialRelationLabels = document.getElementById('relation-labels-toggle')?.checked ?? true;
    const initialHierarchical = document.getElementById('hierarchical-toggle')?.checked ?? false;

    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);

    // DataView (no longer filtering structure nodes)
    nodesView = new vis.DataView(nodesDS, {
        filter: (node) => true
    });

    edgesView = new vis.DataView(edgesDS, {
        filter: (edge) => true
    });


    const data = {
        nodes: nodesView,
        edges: edgesView
    };

    const options = {
        nodes: {
            shape: 'dot',
            size: 20,
            font: {
                size: initialEntityLabels ? 14 : 0,
                color: '#f8fafc',
                face: 'Outfit'
            },
            borderWidth: 2,
            shadow: true,
            color: {
                background: '#6366f1',
                border: '#ffffff',
                highlight: { background: '#818cf8', border: '#ffffff' }
            }
        },
        edges: {
            width: 2,
            color: { color: '#475569', opacity: 0.5, highlight: '#ffffff' },
            smooth: { type: 'continuous', forceDirection: 'none', roundness: 0.5 },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
            font: {
                size: initialRelationLabels ? 12 : 0,
                color: '#e2e8f0',
                face: 'Outfit',
                strokeWidth: 3,
                strokeColor: '#0f172a',
                align: 'middle'
            }
        },
        physics: {
            enabled: initialPhysics && !initialHierarchical,
            stabilization: { iterations: 200 },
            barnesHut: {
                gravitationalConstant: initialGravity,
                centralGravity: 0.1,
                springLength: initialSpringLength,
                springConstant: 0.03,
                damping: 0.08,
                avoidOverlap: 0.5
            }
        },
        layout: {
            hierarchical: {
                enabled: initialHierarchical,
                direction: 'UD',
                sortMethod: 'directed',
                shakeTowards: 'leaves'
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200
        }
    };

    network = new vis.Network(container, data, options);

    // Load initial data
    loadInitialData();
}

async function loadInitialData() {
    try {
        const res = await fetch(`${BACKEND_URL}/graph`);
        const data = await res.json();
        if (data.status === 'success' && data.results) {
            updateGraph(data.results);
        }
    } catch (e) {
        console.error("Failed to load initial graph:", e);
    }
}

// Clear DB
clearDbBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear the entire database? This cannot be undone.')) {
        return;
    }

    clearDbBtn.disabled = true;
    clearDbBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Clearing...';
    lucide.createIcons();

    try {
        const res = await fetch(`${BACKEND_URL}/clear_db`, { method: 'POST' });
        const data = await res.json();

        if (data.status === 'success') {
            nodesDS.clear();
            edgesDS.clear();
            ingestStatus.innerText = 'Database cleared successfully.';
            ingestStatus.className = 'status-msg success';
        } else {
            throw new Error(data.detail || 'Failed to clear database');
        }
    } catch (e) {
        ingestStatus.innerText = `Error clearing DB: ${e.message}`;
        ingestStatus.className = 'status-msg error';
    } finally {
        clearDbBtn.disabled = false;
        clearDbBtn.innerHTML = '<i data-lucide="trash-2"></i> Clear DB';
        lucide.createIcons();
    }
});

// Ingest
ingestBtn.addEventListener('click', async () => {
    const text = ingestText.value;
    if (!text) return;

    ingestBtn.disabled = true;
    ingestBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Processing...';
    lucide.createIcons();
    ingestStatus.innerText = '';
    ingestStatus.className = 'status-msg';
    progressContainer.classList.remove('hidden');


    try {
        const response = await fetch(`${BACKEND_URL}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                clear_db: false
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process SSE formatted data (data: {...}\n\n)
            let lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial last line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));

                    if (data.type === 'progress') {
                        // Switch from indeterminate to functional
                        const progressBar = progressContainer.querySelector('.progress-bar');
                        progressBar.classList.remove('indeterminate');

                        // Calculate percentage for this stage
                        const stagePercent = data.total > 0 ? (data.current / data.total) * 100 : data.progress;
                        progressBar.style.width = `${stagePercent}%`;

                        // Show percentage-based progress
                        let statusText = data.status || 'Processing...';
                        if (data.stage) {
                            const percent = Math.round(data.total > 0 ? (data.current / data.total) * 100 : data.progress);
                            statusText = `${data.stage}: ${percent}%`;
                        }



                        ingestStatus.innerText = statusText;
                    } else if (data.type === 'result') {
                        ingestStatus.innerText = `Success! Processed ${data.results.nodes_processed} document chunks, extracted ${data.results.entities_extracted} entities.`;
                        ingestStatus.classList.add('success');

                        // Refresh whole graph to get new coloring/merges
                        loadInitialData();

                        ingestText.value = '';
                    } else if (data.type === 'error') {
                        throw new Error(data.detail || 'Unknown error');
                    }
                }
            }
        }
    } catch (e) {
        ingestStatus.innerText = `Error: ${e.message}`;
        ingestStatus.classList.add('error');
    } finally {
        ingestBtn.disabled = false;
        ingestBtn.innerHTML = '<i data-lucide="upload-cloud"></i> Ingest';

        // Keep status visible, hide bar after a delay 
        setTimeout(() => {
            progressContainer.classList.add('hidden');
            const progressBar = progressContainer.querySelector('.progress-bar');
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '0%';
        }, 1000);


        lucide.createIcons();
    }


});

// Fit Button
document.getElementById('fit-btn').addEventListener('click', () => {
    if (network) network.fit({ animation: true });
});

// Settings Controls
const springLengthRange = document.getElementById('spring-length-range');
const springLengthVal = document.getElementById('spring-length-val');
const gravityRange = document.getElementById('gravity-range');
const gravityVal = document.getElementById('gravity-val');
const physicsToggle = document.getElementById('physics-toggle');
const entityLabelsToggle = document.getElementById('entity-labels-toggle');
const relationLabelsToggle = document.getElementById('relation-labels-toggle');

springLengthRange.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    springLengthVal.innerText = val;
    network.setOptions({ physics: { barnesHut: { springLength: val } } });
});

gravityRange.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    gravityVal.innerText = val;
    network.setOptions({ physics: { barnesHut: { gravitationalConstant: val } } });
});

physicsToggle.addEventListener('change', (e) => {
    network.setOptions({ physics: { enabled: e.target.checked } });
});

entityLabelsToggle.addEventListener('change', (e) => {
    network.setOptions({
        nodes: {
            font: {
                size: e.target.checked ? 14 : 0
            }
        }
    });
});

relationLabelsToggle.addEventListener('change', (e) => {
    network.setOptions({
        edges: {
            font: {
                size: e.target.checked ? 12 : 0
            }
        }
    });
});

const hierarchicalToggle = document.getElementById('hierarchical-toggle');

hierarchicalToggle.addEventListener('change', (e) => {
    const isHierarchical = e.target.checked;
    network.setOptions({
        layout: {
            hierarchical: {
                enabled: isHierarchical,
                direction: 'UD',
                sortMethod: 'directed',
                shakeTowards: 'leaves'
            }
        },
        physics: {
            enabled: !isHierarchical
        }
    });
    if (isHierarchical) {
        network.stabilize();
    }
});

const COMMUNITY_COLORS = [
    '#ec4899', // Pink
    '#f59e0b', // Amber
    '#10b981', // Emerald
    '#3b82f6', // Blue
    '#8b5cf6', // Violet
    '#ef4444', // Red
    '#06b6d4', // Cyan
    '#84cc16'  // Lime
];

function updateGraph(results) {
    const { entities, relations } = results;

    // Add Entities
    entities.forEach(ent => {
        if (ent.type === 'QueryEntity') return;

        // Coloring based on community
        let nodeColor = '#6366f1'; // Default
        if (ent.community_id !== undefined && ent.community_id !== null && ent.community_id !== -1) {
            nodeColor = COMMUNITY_COLORS[ent.community_id % COMMUNITY_COLORS.length];
        }

        try {
            nodesDS.update({
                id: ent.id,
                label: ent.name || ent.metadata?.name || ent.id,
                type: ent.type,
                title: `Type: ${ent.type}\nCommunity: ${ent.community_id ?? 'None'}\nDesc: ${ent.description || ''}`,
                color: {
                    background: nodeColor,
                    border: '#ffffff',
                    highlight: { background: nodeColor, border: '#ffffff' }
                }
            });
        } catch (e) { }
    });

    // Add Relations
    relations.forEach(rel => {
        const edgeId = `${rel.source_id}-${rel.type}-${rel.target_id}`;
        try {
            edgesDS.update({
                id: edgeId,
                from: rel.source_id,
                to: rel.target_id,
                label: rel.type,
                title: rel.description,
                hidden: false
            });
        } catch (e) { }
    });

    network.fit({ animation: true });
}

// Query
queryBtn.addEventListener('click', async () => {
    const query = queryInput.value;
    if (!query) return;

    queryBtn.disabled = true;
    queryBtn.innerText = 'Searching...';
    queryResults.innerHTML = '';

    try {
        const res = await fetch(`${BACKEND_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                query_type: 'local'
            })
        });

        const data = await res.json();

        if (data.status === 'success') {
            renderAnswer(data.results);
            updateGraph(data.results);
        } else {
            queryResults.innerHTML = `<div class="status-msg error">Error: ${data.detail}</div>`;
        }

    } catch (e) {
        queryResults.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
    } finally {
        queryBtn.disabled = false;
        queryBtn.innerText = 'Run Query';
    }
});

function renderAnswer(results) {
    queryResults.innerHTML = '';

    // Show Answer
    if (results.answer) {
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer-box';
        answerDiv.innerHTML = `<strong>Answer:</strong><br>${results.answer.replace(/\n/g, '<br>')}`;
        queryResults.appendChild(answerDiv);
    }

    // Show Context (optional, or just rely on graph)
    if (results.relations && results.relations.length > 0) {
        const contextHeader = document.createElement('div');
        contextHeader.className = 'subtitle';
        contextHeader.style.marginTop = '1rem';
        contextHeader.innerText = 'Context Sources:';
        queryResults.appendChild(contextHeader);

        results.relations.forEach(rel => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <div class="result-relation">${rel.source_name || rel.source_id} &rarr; ${rel.target_name || rel.target_id}</div>
                <div class="result-text">${rel.description}</div>
            `;
            queryResults.appendChild(card);
        });
    } else if (!results.answer) {
        queryResults.innerHTML = '<div class="empty-state">No results found.</div>';
    }
}


// Add spinning animation style for loader
const style = document.createElement('style');
style.innerHTML = `
@keyframes spin { 100% { transform: rotate(360deg); } }
.spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);
