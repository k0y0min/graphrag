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

const queryBtn = document.getElementById('query-btn');
const queryInput = document.getElementById('query-input');
const queryResults = document.getElementById('query-results');
const showStructureCheck = document.getElementById('show-structure-check');

const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Init
document.addEventListener('DOMContentLoaded', () => {
    initGraph();
});

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

    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);

    // DataView for filtering
    nodesView = new vis.DataView(nodesDS, {
        filter: (node) => {
            const isStructure = ['Structure_Header', 'Structure_Content', 'content', 'header'].includes(node.type);
            if (isStructure && !showStructureCheck.checked) {
                return false;
            }
            return true;
        }
    });

    // edges should also be filtered if their endpoints are hidden? 
    // Vis network handles that automatically usually, but let's be safe.
    edgesView = new vis.DataView(edgesDS, {
        filter: (edge) => {
            const fromNode = nodesDS.get(edge.from);
            const toNode = nodesDS.get(edge.to);
            if (!fromNode || !toNode) return false;

            const isFromStructure = ['Structure_Header', 'Structure_Content', 'content', 'header'].includes(fromNode.type);
            const isToStructure = ['Structure_Header', 'Structure_Content', 'content', 'header'].includes(toNode.type);

            if (!showStructureCheck.checked && (isFromStructure || isToStructure)) {
                return false;
            }
            return true;
        }
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
                size: 14,
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
            color: { color: '#94a3b8', highlight: '#ffffff' },
            smooth: { type: 'continuous' },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } }
        },
        physics: {
            stabilization: false,
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 95,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0
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
    console.log("Fetching initial graph data...");
    try {
        const res = await fetch('/graph');
        const data = await res.json();
        console.log("Graph data received:", data);
        if (data.status === 'success' && data.results) {
            updateGraphFromIngest(data.results);
            console.log("Graph updated with initial data.");
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
        const res = await fetch('/clear_db', { method: 'POST' });
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
    lucide.createIcons(); // Refresh icons
    ingestStatus.innerText = 'Extracting entities & building graph...';
    ingestStatus.className = 'status-msg';

    try {
        const res = await fetch('/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                clear_db: false // Always false now as we have a dedicated button
            })
        });

        const data = await res.json();

        if (data.status === 'success') {
            ingestStatus.innerText = `Success! Processed ${data.results.nodes_processed} document chunks, extracted ${data.results.entities_extracted} entities.`;
            ingestStatus.classList.add('success');

            if (data.results.entities && data.results.relations) {
                updateGraphFromIngest(data.results);
            }

            ingestText.value = '';
        } else {
            throw new Error(data.detail || 'Unknown error');
        }
    } catch (e) {
        ingestStatus.innerText = `Error: ${e.message}`;
        ingestStatus.classList.add('error');
    } finally {
        ingestBtn.disabled = false;
        ingestBtn.innerHTML = '<i data-lucide="upload-cloud"></i> Ingest';
        lucide.createIcons();
    }
});

// Fit Button
document.getElementById('fit-btn').addEventListener('click', () => {
    if (network) network.fit({ animation: true });
});

// Structure Toggle
showStructureCheck.addEventListener('change', () => {
    updateNodeVisibility();
    // Refresh views if necessary (DataView usually handles this, but let's be explicit)
    nodesView.refresh();
    edgesView.refresh();
    if (network) network.fit({ animation: true });
});

function updateNodeVisibility() {
    const showStructure = showStructureCheck.checked;

    // DataView filter already handles the logic, 
    // but sometimes explicit refresh or hidden property update helps.
    // Since we're using DataView, we just need to refresh it.
    nodesView.refresh();
    edgesView.refresh();
}

function updateGraphFromIngest(results) {
    const { entities, relations } = results;

    // Add Entities
    entities.forEach(ent => {
        try {
            nodesDS.update({
                id: ent.id,
                label: ent.id,
                type: ent.type, // Store type for filtering
                title: `Type: ${ent.type}\nDesc: ${ent.description || ''}`,
                color: ent.type === 'Structure_Header' ? '#ef4444' : '#6366f1'
            });
        } catch (e) { }
    });

    // Add Relations
    relations.forEach(rel => {
        const edgeId = `${rel.source_id}-${rel.type}-${rel.target_id}`;
        try {
            const isParentOf = rel.type === 'PARENT_OF';
            edgesDS.update({
                id: edgeId,
                from: rel.source_id,
                to: rel.target_id,
                label: rel.type,
                title: rel.description,
                hidden: isParentOf ? !showStructureCheck.checked : false
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
        const res = await fetch('/query', {
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
            updateGraphFromQuery(data.results, query);
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
                <div class="result-relation">${rel.source_id} &rarr; ${rel.target_id}</div>
                <div class="result-text">${rel.description}</div>
            `;
            queryResults.appendChild(card);
        });
    } else if (!results.answer) {
        queryResults.innerHTML = '<div class="empty-state">No results found.</div>';
    }
}

function updateGraphFromQuery(results, queryText) {
    const { entities, relations } = results;

    if (!entities || entities.length === 0) {
        return;
    }

    // Add Entities
    entities.forEach(ent => {
        try {
            // Remove QueryEntity logic - we only show real matches from DB
            if (ent.type === 'QueryEntity') return;

            nodesDS.update({
                id: ent.id,
                label: ent.id,
                type: ent.type,
                title: `Type: ${ent.type}\nDesc: ${ent.description || ''}`,
                color: '#6366f1'
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
                title: rel.description
            });
        } catch (e) { }
    });

    network.fit({ animation: true });
}

// Add spinning animation style for loader
const style = document.createElement('style');
style.innerHTML = `
@keyframes spin { 100% { transform: rotate(360deg); } }
.spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);
