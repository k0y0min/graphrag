// Backend configuration
const BACKEND_URL = window.BACKEND_URL || '/api';
const POLL_INTERVAL = 2000; // 2 seconds

// Auth globals
let authToken = localStorage.getItem('graphrag_token');
let authMode = 'login'; // 'login' or 'register'

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

// Auth DOM
const authOverlay = document.getElementById('auth-overlay');
const authTabLogin = document.getElementById('auth-tab-login');
const authTabRegister = document.getElementById('auth-tab-register');
const authSubmitBtn = document.getElementById('auth-submit-btn');
const authUsername = document.getElementById('auth-username');
const authPassword = document.getElementById('auth-password');
const authError = document.getElementById('auth-error');

// API Wrapper
async function fetchApi(endpoint, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    
    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        ...options,
        headers
    });
    
    if (response.status === 401 && endpoint !== '/auth/login') {
        logout();
        throw new Error("Unauthorized. Please log in.");
    }
    return response;
}

function logout() {
    authToken = null;
    localStorage.removeItem('graphrag_token');
    authOverlay.classList.remove('hidden');
    if (network) {
        nodesDS.clear();
        edgesDS.clear();
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    setupAuthListeners();
    checkBackendReady();
});

// Auth Setup
function setupAuthListeners() {
    authTabLogin.addEventListener('click', () => {
        authMode = 'login';
        authTabLogin.classList.add('active');
        authTabRegister.classList.remove('active');
        authSubmitBtn.innerText = 'Sign In';
        authError.innerText = '';
    });
    
    authTabRegister.addEventListener('click', () => {
        authMode = 'register';
        authTabRegister.classList.add('active');
        authTabLogin.classList.remove('active');
        authSubmitBtn.innerText = 'Create Account';
        authError.innerText = '';
    });
    
    authSubmitBtn.addEventListener('click', async () => {
        const username = authUsername.value.trim();
        const password = authPassword.value;
        if (!username || !password) {
            authError.innerText = "Username and password required.";
            return;
        }
        
        authSubmitBtn.disabled = true;
        authSubmitBtn.innerText = 'Processing...';
        
        try {
            if (authMode === 'register') {
                const res = await fetch(`${BACKEND_URL}/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Registration failed');
                
                // Auto-login after register
                authTabLogin.click();
                authSubmitBtn.innerText = 'Registration successful! Click Sign In.';
            } else {
                const res = await fetch(`${BACKEND_URL}/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Login failed');
                
                authToken = data.access_token;
                localStorage.setItem('graphrag_token', authToken);
                
                authOverlay.classList.add('hidden');
                initGraph();
            }
        } catch (e) {
            authError.innerText = e.message;
        } finally {
            authSubmitBtn.disabled = false;
            if (authMode === 'login') authSubmitBtn.innerText = 'Sign In';
            if (authMode === 'register') authSubmitBtn.innerText = 'Create Account';
        }
    });

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            logout();
        });
    }
}

function showEphemeralToast(message, duration = 5000) {
    let toast = document.getElementById('ephemeral-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'ephemeral-toast';
        toast.style.cssText = "position: fixed; bottom: 20px; right: 20px; background: #1e293b; color: white; padding: 12px 24px; border-radius: 8px; z-index: 100000; font-family: 'Outfit', sans-serif; font-size: 0.95rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); transition: opacity 0.5s ease; max-width: 350px; border-left: 4px solid #fbbf24;";
        document.body.appendChild(toast);
    }
    toast.innerText = message;
    toast.style.opacity = '1';
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => { toast.style.display = 'none'; }, 500);
    }, duration);
}

async function checkBackendReady() {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // Hide loading overlay immediately to let user interact
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
    
    if (authToken) {
        initGraph();
    } else {
        authOverlay.classList.remove('hidden');
    }

    // Start background polling
    pollBackendStatus();
}

async function pollBackendStatus() {
    let shownWarmingUpMessage = false;
    
    while (true) {
        try {
            const res = await fetch(`${BACKEND_URL}/health`);
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'healthy') {
                    console.log("Backend is fully healthy!");
                    break; // Stop polling
                } else if (data.status === 'warming_up') {
                    if (!shownWarmingUpMessage) {
                        showEphemeralToast("vLLM is warming up (takes ~5 mins). Perplexity chunking is inactive; falling back to fixed-size chunking for now.", 6000);
                        shownWarmingUpMessage = true;
                    }
                }
            }
        } catch (e) {
            console.log("Waiting for backend...");
        }
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
}

// Tabs
tabBtns.forEach(btn => {
    // Only process if it's not an auth tab
    if (btn.id.startsWith('auth-tab')) return;
    
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => {
            if (!b.id.startsWith('auth-tab')) b.classList.remove('active');
        });
        tabContents.forEach(c => c.classList.add('hidden'));

        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`${tabId}-tab`).classList.remove('hidden');
    });
});

// Graph Initialization
function initGraph() {
    if (network) return; // Already initialized
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

    nodesView = new vis.DataView(nodesDS, { filter: (node) => true });
    edgesView = new vis.DataView(edgesDS, { filter: (edge) => true });

    const data = { nodes: nodesView, edges: edgesView };

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
        interaction: { hover: true, tooltipDelay: 200 }
    };

    network = new vis.Network(container, data, options);
    loadInitialData();
}

async function loadInitialData() {
    try {
        const res = await fetchApi(`/graph`);
        const data = await res.json();
        if (data.status === 'success' && data.results) {
            nodesDS.clear();
            edgesDS.clear();
            updateGraph(data.results);
        }
    } catch (e) {
        console.error("Failed to load initial graph:", e);
    }
}

// Clear DB
clearDbBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear your graph database? This cannot be undone.')) {
        return;
    }

    clearDbBtn.disabled = true;
    clearDbBtn.innerHTML = 'Clearing...';

    try {
        const res = await fetchApi(`/clear_db`, { method: 'POST' });
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
        ingestStatus.innerText = `Error: ${e.message}`;
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
    ingestBtn.innerHTML = '<i data-lucide="loader-2" class="spin" style="position: absolute; left: 1rem;"></i> <span>Processing...</span>';
    lucide.createIcons();
    ingestStatus.innerText = '';
    ingestStatus.className = 'status-msg';
    progressContainer.classList.remove('hidden');

    try {
        const response = await fetchApi(`/ingest`, {
            method: 'POST',
            body: JSON.stringify({ text: text, clear_db: false })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));

                    if (data.type === 'progress') {
                        const progressBar = progressContainer.querySelector('.progress-bar');
                        progressBar.classList.remove('indeterminate');

                        const stagePercent = data.total > 0 ? (data.current / data.total) * 100 : data.progress;
                        progressBar.style.width = `${stagePercent}%`;

                        let statusText = data.status || 'Processing...';
                        if (data.stage) {
                            const percent = Math.round(data.total > 0 ? (data.current / data.total) * 100 : data.progress);
                            statusText = `${data.stage}: ${percent}%`;
                        }
                        ingestStatus.innerText = statusText;
                    } else if (data.type === 'result') {
                        ingestStatus.innerText = `Success! Processed ${data.results.nodes_processed} chunks, extracted ${data.results.entities_extracted} entities.`;
                        ingestStatus.classList.add('success');
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
document.getElementById('spring-length-range').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('spring-length-val').innerText = val;
    network.setOptions({ physics: { barnesHut: { springLength: val } } });
});

document.getElementById('gravity-range').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('gravity-val').innerText = val;
    network.setOptions({ physics: { barnesHut: { gravitationalConstant: val } } });
});

document.getElementById('physics-toggle').addEventListener('change', (e) => {
    network.setOptions({ physics: { enabled: e.target.checked } });
});

document.getElementById('entity-labels-toggle').addEventListener('change', (e) => {
    network.setOptions({ nodes: { font: { size: e.target.checked ? 14 : 0 } } });
});

document.getElementById('relation-labels-toggle').addEventListener('change', (e) => {
    network.setOptions({ edges: { font: { size: e.target.checked ? 12 : 0 } } });
});

document.getElementById('hierarchical-toggle').addEventListener('change', (e) => {
    const isHierarchical = e.target.checked;
    network.setOptions({
        layout: { hierarchical: { enabled: isHierarchical, direction: 'UD', sortMethod: 'directed' } },
        physics: { enabled: !isHierarchical }
    });
    if (isHierarchical) network.stabilize();
});

const COMMUNITY_COLORS = ['#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ef4444', '#06b6d4', '#84cc16'];

function updateGraph(results) {
    const { entities, relations } = results;

    entities.forEach(ent => {
        if (ent.type === 'QueryEntity') return;

        let nodeColor = '#6366f1';
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
        const res = await fetchApi(`/query`, {
            method: 'POST',
            body: JSON.stringify({ query: query, query_type: 'local' })
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

    if (results.answer) {
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer-box';
        answerDiv.innerHTML = `<strong>Answer:</strong><br>${results.answer.replace(/\n/g, '<br>')}`;
        queryResults.appendChild(answerDiv);
    }

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
