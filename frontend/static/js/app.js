// ==========================================================================
// GRAPHRAG STUDIO ENTERPRISE CLIENT - NEO4J BLOOM / MEMGRAPH LAB STYLE
// Full-Screen Knowledge Graph Canvas, Floating Omni-Bar & Deep Inspection
// ==========================================================================

const BACKEND_URL = window.BACKEND_URL || '/api';

// Global Application State
let authToken = localStorage.getItem('graphrag_token');
let authMode = 'login';
let omniMode = 'query'; // 'query' | 'ingest'
let attachedFile = null;
let network = null;
let nodesDS = null;
let edgesDS = null;
let nodesView = null;
let edgesView = null;
let latestChunks = [];
let latestAliases = [];
let latestQueryTrace = null;

// Filter & Sub-Graph Scope State
let activeFilter = {
    community: 'all',
    selectedCommunities: new Set(),
    entityType: 'all',
    selectedNodeId: null,
    hopDepth: 'all'
};
let activeNeighborhoodIds = null;

// Community Color Palette (Vibrant Cyber Palette)
const COMMUNITY_COLORS = [
    '#6366f1', // Electric Indigo
    '#06b6d4', // Cyber Cyan
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#ec4899', // Pink
    '#8b5cf6', // Violet
    '#f43f5e', // Crimson
    '#14b8a6', // Teal
    '#3b82f6', // Cobalt
    '#84cc16'  // Lime
];

// ==========================================================================
// Mutually Exclusive Panel / Drawer Management
// ==========================================================================
function closeAllFloatingPanels(exceptId = null) {
    const panels = [
        { id: 'filter-dock', el: document.getElementById('filter-dock') },
        { id: 'history-drawer', el: document.getElementById('history-drawer') },
        { id: 'bloom-inspector', el: document.getElementById('bloom-inspector') }
    ];
    panels.forEach(p => {
        if (p.el && p.id !== exceptId) {
            if (p.id === 'history-drawer' && localStorage.getItem('graphrag_history_sidebar_pinned') === 'true') {
                return; // Leave history sidebar pinned open if frozen
            }
            p.el.classList.add('hidden');
            if (p.id === 'history-drawer') {
                const sbBtn = document.getElementById('sidebar-toggle-btn');
                if (sbBtn) sbBtn.classList.remove('hidden');
            }
        }
    });
}

// ==========================================================================
// History & Persistence Management
// ==========================================================================
const SESSIONS_STORAGE_KEY = 'graphrag_sessions_history';
const ACTIVE_SESSION_STORAGE_KEY = 'graphrag_active_session';

function getSavedSessions() {
    try {
        const raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        console.error("Failed to parse saved sessions:", e);
        return [];
    }
}

function saveSessionRecord(session) {
    try {
        const sessions = getSavedSessions();
        const updated = [session, ...sessions.filter(s => s.id !== session.id)].slice(0, 40);
        localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(updated));
        localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, JSON.stringify(session));
        renderSessionsDrawer();
    } catch (e) {
        console.error("Failed to save session:", e);
    }
}

function renderSessionsDrawer() {
    const container = document.getElementById('sessions-list');
    if (!container) return;

    const sessions = getSavedSessions();
    if (sessions.length === 0) {
        container.innerHTML = '<div class="empty-state">No activity recorded yet. Ingest documents or run queries to see history here.</div>';
        return;
    }

    container.innerHTML = '';
    sessions.forEach(sess => {
        const card = document.createElement('div');
        card.className = 'session-card-item';
        card.setAttribute('data-id', sess.id);

        const badgeClass = sess.type === 'query' ? 'tag-query' : 'tag-ingest';
        const badgeText = sess.type === 'query' ? (sess.trace?.route?.toUpperCase() || 'QUERY') : 'INGEST';

        card.innerHTML = `
            <div class="session-card-top">
                <span class="session-badge-tag ${badgeClass}">${badgeText}</span>
                <span>${sess.timestamp || ''}</span>
            </div>
            <div class="session-card-title">${sess.title || 'Activity Record'}</div>
            <div class="session-card-snippet">${sess.snippet || ''}</div>
        `;

        card.addEventListener('click', () => {
            document.querySelectorAll('.session-card-item').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            loadSessionIntoView(sess);
        });

        container.appendChild(card);
    });
}

function loadSessionIntoView(session) {
    latestChunks = session.chunks || [];
    latestAliases = session.aliases || [];
    latestQueryTrace = session.trace || null;

    if (typeof renderChunksGrid === 'function') renderChunksGrid(latestChunks);
    if (typeof renderAliasesTable === 'function') renderAliasesTable(latestAliases);
    if (typeof renderLangGraphTrace === 'function') renderLangGraphTrace(latestQueryTrace);

    const expandCard = document.getElementById('omni-expand-card');
    const answerEl = document.getElementById('omni-answer-content');
    const modeBadge = document.getElementById('expand-mode-badge');
    const groundedBadge = document.getElementById('expand-grounded-badge');

    if (expandCard && answerEl) {
        expandCard.classList.remove('hidden');
        if (window.marked && session.answer) {
            answerEl.innerHTML = `<div class="query-answer-body">${marked.parse(session.answer)}</div>`;
        } else {
            answerEl.innerHTML = `<div class="query-answer-body"><p>${session.answer || ''}</p></div>`;
        }

        if (modeBadge) {
            if (session.type === 'query') {
                const route = session.trace?.route || 'hybrid';
                modeBadge.innerText = route === 'graph' ? '🧭 Cypher Graph Traversal' : (route === 'vector' ? '📄 Contextual Text Vector' : '🔍 Multi-Hop Hybrid Retrieval');
            } else {
                modeBadge.innerText = '📦 Ingestion Summary';
            }
        }

        if (groundedBadge) {
            if (session.type === 'query' && session.trace) {
                groundedBadge.classList.remove('hidden');
                groundedBadge.innerHTML = session.trace.is_grounded !== false ? '<i data-lucide="shield-check"></i> Grounded' : '<i data-lucide="alert-triangle"></i> Partial';
            } else {
                groundedBadge.classList.add('hidden');
            }
        }

        const timerEl = document.getElementById('pipeline-timer');
        if (timerEl) {
            timerEl.innerText = session.duration ? `${session.duration}s` : 'Saved';
        }

        const statusText = document.getElementById('pipeline-status-text');
        if (statusText) {
            if (session.type === 'query') {
                statusText.innerText = "Grounded response synthesized across knowledge graph and passages.";
            } else {
                statusText.innerText = session.snippet || "Indexed knowledge graph entities, relations, and communities into LadybugDB.";
            }
        }

        if (typeof pipelineStepper !== 'undefined' && pipelineStepper && typeof pipelineStepper.renderCompletedStages === 'function') {
            pipelineStepper.renderCompletedStages(session.type === 'query' ? 'query' : 'ingest');
        }

        if (window.lucide) lucide.createIcons();
    }

    if (session.trace?.extracted_entities && network && nodesDS) {
        const ents = session.trace.extracted_entities;
        const matchingIds = [];
        nodesDS.forEach(n => {
            if (ents.some(e => (n.label || '').toLowerCase().includes(e.toLowerCase()))) {
                matchingIds.push(n.id);
            }
        });
        if (matchingIds.length > 0) {
            network.selectNodes(matchingIds);
        }
    }
}

function restoreActiveSession() {
    try {
        const raw = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
        if (raw) {
            const sess = JSON.parse(raw);
            loadSessionIntoView(sess);
        }
    } catch (e) {
        console.warn("Could not restore active session:", e);
    }
}

// Sample Technical Datasets
const SAMPLE_DATASETS = {
    starship: `SpaceX is developing the Starship launch system, which consists of the Super Heavy booster (Stage 1) and the Starship spacecraft (Stage 2). 

On June 6, 2024, SpaceX conducted Integrated Flight Test 4 (IFT-4) from Starbase in Boca Chica, Texas. Super Heavy successfully executed a soft splashdown in the Gulf of Mexico after separating from Starship via hot-staging. Starship survived atmospheric reentry despite extreme heat and flap erosion, executing a controlled landing burn in the Indian Ocean.

Elon Musk founded SpaceX in 2002 to reduce space transportation costs and enable the colonization of Mars. Gwen Shotwell serves as SpaceX President and Chief Operating Officer, managing daily operations and commercial satellite launches. SpaceX utilizes Raptor engines fueled by liquid methane and liquid oxygen (methalox), manufactured in McGregor, Texas.`,

    apollo: `The Apollo 11 mission was the historic spaceflight that first landed humans on the Moon on July 20, 1969. 

Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on the Sea of Tranquility. Armstrong became the first person to walk on the lunar surface, famously declaring: "That's one small step for man, one giant leap for mankind."

Command Module Pilot Michael Collins flew the Command/Service Module Columbia alone in lunar orbit while his crewmates were on the surface. Apollo 11 was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, Florida. NASA Administrator Thomas O. Paine oversaw the program following the vision set forth by President John F. Kennedy in 1961.`,

    'ai-agents': `Modern agentic artificial intelligence systems combine Large Language Models (LLMs) with Graph Retrieval-Augmented Generation (GraphRAG) and symbolic execution.

LangGraph is an orchestration framework developed by Harrison Chase and the LangChain team for building stateful, multi-agent workflows with cycles and self-correction. In our architecture, LangGraph routes queries between vector text search and Cypher graph traversals in LadybugDB.

LadybugDB is an embedded property graph database supporting the Cypher query language with ACID compliance and zero external service overhead. Gemini 3.8-Flash from Google DeepMind powers real-time schema-enforced entity extraction, Macro-Context Horizon indexing, and hallucination self-grading.`
};

// ==========================================================================
// Initialization & Authentication
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    setupAuth();
    setupTopBar();
    setupOmniBar();
    setupFullWindowDrop();
    setupFilterDock();
    setupCanvasHUD();
    setupInspectorDrawer();
    setupDeepInspectionModal();
    setupCypherConsole();
    checkBackendReady();
});

// Generic Fetch Wrapper with Auth Header
async function fetchApi(endpoint, options = {}) {
    const headers = { ...options.headers };
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }

    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        ...options,
        headers
    });

    if (response.status === 401 && endpoint !== '/auth/login' && endpoint !== '/auth/register') {
        logout();
        throw new Error("Session expired. Please log in again.");
    }
    return response;
}

function setupAuth() {
    const authOverlay = document.getElementById('auth-overlay');
    const authTabLogin = document.getElementById('auth-tab-login');
    const authTabRegister = document.getElementById('auth-tab-register');
    const authSubmitBtn = document.getElementById('auth-submit-btn');
    const authUsername = document.getElementById('auth-username');
    const authPassword = document.getElementById('auth-password');
    const authError = document.getElementById('auth-error');
    const logoutBtn = document.getElementById('logout-btn');

    if (!authOverlay) return;

    authTabLogin.addEventListener('click', () => {
        authMode = 'login';
        authTabLogin.classList.add('active');
        authTabRegister.classList.remove('active');
        authSubmitBtn.querySelector('span').innerText = 'Sign In';
        authError.classList.add('hidden');
    });

    authTabRegister.addEventListener('click', () => {
        authMode = 'register';
        authTabRegister.classList.add('active');
        authTabLogin.classList.remove('active');
        authSubmitBtn.querySelector('span').innerText = 'Create Account';
        authError.classList.add('hidden');
    });

    authSubmitBtn.addEventListener('click', async () => {
        const username = authUsername.value.trim();
        const password = authPassword.value;
        if (!username || !password) {
            showAuthError("Username and password are required.");
            return;
        }

        authSubmitBtn.disabled = true;

        try {
            if (authMode === 'register') {
                const res = await fetch(`${BACKEND_URL}/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Registration failed');

                showToast("Account created successfully. Signing in...", "success");
                authTabLogin.click();
            }

            // Perform Login
            const res = await fetch(`${BACKEND_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Login failed');

            authToken = data.access_token;
            localStorage.setItem('graphrag_token', authToken);
            localStorage.setItem('graphrag_username', username);
            const userAvatar = document.getElementById('user-avatar-initial');
            if (userAvatar) userAvatar.innerText = username.charAt(0).toUpperCase();
            const userNameDisplay = document.getElementById('user-name-display');
            if (userNameDisplay) userNameDisplay.innerText = username;

            authOverlay.classList.add('hidden');
            showToast(`Welcome back, ${username}!`, "info");
            initGraph();
            loadGraphData();
            refreshStats();
        } catch (e) {
            showAuthError(e.message);
        } finally {
            authSubmitBtn.disabled = false;
        }
    });

    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            if (confirm("Sign out of GraphRAG Studio?")) {
                logout();
            }
        });
    }

    function showAuthError(msg) {
        authError.innerText = msg;
        authError.classList.remove('hidden');
    }
}

function logout() {
    authToken = null;
    localStorage.removeItem('graphrag_token');
    const authOverlay = document.getElementById('auth-overlay');
    if (authOverlay) authOverlay.classList.remove('hidden');
    if (network) {
        network.destroy();
        network = null;
        nodesDS = null;
        edgesDS = null;
    }
}

async function checkBackendReady() {
    const loadingOverlay = document.getElementById('loading-overlay');
    while (true) {
        try {
            const res = await fetch(`${BACKEND_URL}/health`);
            if (res.ok) break;
        } catch (e) {
            // Keep polling
        }
        await new Promise(r => setTimeout(r, 600));
    }

    if (loadingOverlay) loadingOverlay.classList.add('hidden');

    if (authToken) {
        const username = localStorage.getItem('graphrag_username') || 'demo';
        const userAvatar = document.getElementById('user-avatar-initial');
        if (userAvatar) userAvatar.innerText = username.charAt(0).toUpperCase();
        const userNameDisplay = document.getElementById('user-name-display');
        if (userNameDisplay) userNameDisplay.innerText = username;

        initGraph();
        loadGraphData();
        refreshStats();
        restoreActiveSession();
        renderSessionsDrawer();
    } else {
        const authOverlay = document.getElementById('auth-overlay');
        if (authOverlay) authOverlay.classList.remove('hidden');
    }
}

// ==========================================================================
// Top Command Bar Actions
// ==========================================================================
function setupTopBar() {
    const clearDbBtn = document.getElementById('clear-db-btn');
    if (clearDbBtn) {
        clearDbBtn.addEventListener('click', async () => {
            if (!confirm('Clear all graph data?')) {
                return;
            }
            try {
                const res = await fetchApi('/clear_db', { method: 'POST' });
                if (res.ok) {
                    if (nodesDS) nodesDS.clear();
                    if (edgesDS) edgesDS.clear();
                    latestChunks = [];
                    latestAliases = [];
                    localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
                    refreshStats();
                    showToast("Database successfully cleared.", "success");
                }
            } catch (e) {
                showToast(`Clear failed: ${e.message}`, "error");
            }
        });
    }

    const filterToggleBtn = document.getElementById('filter-toggle-btn');
    const filterDock = document.getElementById('filter-dock');
    if (filterToggleBtn && filterDock) {
        filterToggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const willOpen = filterDock.classList.contains('hidden');
            closeAllFloatingPanels();
            if (willOpen) {
                filterDock.classList.remove('hidden');
            }
        });
    }

    const closeFilterDockBtn = document.getElementById('close-filter-dock-btn');
    if (closeFilterDockBtn && filterDock) {
        closeFilterDockBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            filterDock.classList.add('hidden');
        });
    }

    // Activity History Floating Sidebar Handlers (Modern AI Chat Platform Style)
    const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
    const pinHistoryBtn = document.getElementById('pin-history-sidebar-btn');
    const closeHistoryBtn = document.getElementById('close-history-drawer-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyDrawer = document.getElementById('history-drawer');
    const HISTORY_PINNED_KEY = 'graphrag_history_sidebar_pinned';

    function isHistoryPinned() {
        return localStorage.getItem(HISTORY_PINNED_KEY) === 'true';
    }

    function syncHistorySidebarState() {
        if (!historyDrawer) return;
        if (isHistoryPinned()) {
            historyDrawer.classList.add('pinned');
            historyDrawer.classList.remove('hidden');
            if (pinHistoryBtn) pinHistoryBtn.classList.add('pinned');
            if (sidebarToggleBtn) sidebarToggleBtn.classList.add('hidden');
            renderSessionsDrawer();
        } else {
            historyDrawer.classList.remove('pinned');
            if (pinHistoryBtn) pinHistoryBtn.classList.remove('pinned');
            if (!historyDrawer.classList.contains('hidden')) {
                if (sidebarToggleBtn) sidebarToggleBtn.classList.add('hidden');
            } else {
                if (sidebarToggleBtn) sidebarToggleBtn.classList.remove('hidden');
            }
        }
    }

    syncHistorySidebarState();

    if (sidebarToggleBtn && historyDrawer) {
        sidebarToggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            closeAllFloatingPanels('history-drawer');
            historyDrawer.classList.remove('hidden');
            sidebarToggleBtn.classList.add('hidden');
            renderSessionsDrawer();
        });
    }

    if (pinHistoryBtn && historyDrawer) {
        pinHistoryBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const pinned = !isHistoryPinned();
            localStorage.setItem(HISTORY_PINNED_KEY, pinned ? 'true' : 'false');
            syncHistorySidebarState();
            showToast(pinned ? "History sidebar pinned (frozen open)" : "History sidebar unpinned", "info");
        });
    }

    if (closeHistoryBtn && historyDrawer) {
        closeHistoryBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            localStorage.setItem(HISTORY_PINNED_KEY, 'false');
            historyDrawer.classList.add('hidden');
            historyDrawer.classList.remove('pinned');
            if (pinHistoryBtn) pinHistoryBtn.classList.remove('pinned');
            if (sidebarToggleBtn) sidebarToggleBtn.classList.remove('hidden');
        });
    }

    // Edge Hover Detection for History Sidebar (reveal when cursor moves to left edge)
    let hoverHideTimeout = null;

    document.addEventListener('mousemove', (e) => {
        if (isHistoryPinned()) return;

        // When cursor is within 24px of left screen edge (between top bar and bottom omnibar)
        if (e.clientX <= 24 && e.clientY > 60 && e.clientY < window.innerHeight - 70) {
            if (hoverHideTimeout) {
                clearTimeout(hoverHideTimeout);
                hoverHideTimeout = null;
            }
            if (historyDrawer && historyDrawer.classList.contains('hidden')) {
                closeAllFloatingPanels('history-drawer');
                historyDrawer.classList.remove('hidden');
                if (sidebarToggleBtn) sidebarToggleBtn.classList.add('hidden');
                renderSessionsDrawer();
            }
        } else if (historyDrawer && !historyDrawer.classList.contains('hidden')) {
            // If cursor moves outside the sidebar + buffer zone
            if (e.clientX > 350 || e.clientY < 45 || e.clientY > window.innerHeight - 45) {
                if (!hoverHideTimeout) {
                    hoverHideTimeout = setTimeout(() => {
                        if (!isHistoryPinned() && historyDrawer && !historyDrawer.classList.contains('hidden')) {
                            historyDrawer.classList.add('hidden');
                            if (sidebarToggleBtn) sidebarToggleBtn.classList.remove('hidden');
                        }
                        hoverHideTimeout = null;
                    }, 400);
                }
            } else {
                // Cursor is inside sidebar or toggle, cancel hide
                if (hoverHideTimeout) {
                    clearTimeout(hoverHideTimeout);
                    hoverHideTimeout = null;
                }
            }
        }
    });

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm("Clear all activity history?")) {
                localStorage.removeItem(SESSIONS_STORAGE_KEY);
                localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
                renderSessionsDrawer();
                showToast("Activity history cleared.", "info");
            }
        });
    }

    // Modern UX: Close open floating panels when clicking outside
    document.addEventListener('pointerdown', (e) => {
        const target = e.target;

        // 1. Filter Dock: close if open and click is outside dock and button
        if (filterDock && !filterDock.classList.contains('hidden')) {
            if (!filterDock.contains(target) && filterToggleBtn && !filterToggleBtn.contains(target)) {
                filterDock.classList.add('hidden');
            }
        }

        // 2. History Sidebar: close if open and NOT pinned, and click is outside sidebar and toggle button
        if (historyDrawer && !historyDrawer.classList.contains('hidden') && !isHistoryPinned()) {
            if (!historyDrawer.contains(target) && (!sidebarToggleBtn || !sidebarToggleBtn.contains(target))) {
                historyDrawer.classList.add('hidden');
                if (sidebarToggleBtn) sidebarToggleBtn.classList.remove('hidden');
            }
        }

        // 3. Bloom Inspector Drawer: close if open and click is outside drawer and network canvas
        const bloomInspector = document.getElementById('bloom-inspector');
        const networkContainer = document.getElementById('network-container');
        if (bloomInspector && !bloomInspector.classList.contains('hidden')) {
            if (!bloomInspector.contains(target) && networkContainer && !networkContainer.contains(target)) {
                closeNodeInspector();
            }
        }
    });
}

// ==========================================================================
// Live Paced Pipeline Stepper
// ==========================================================================
class PacedPipelineStepper {
    constructor() {
        this.expandCard = document.getElementById('omni-expand-card');
        this.stepperStrip = document.getElementById('pipeline-stepper-strip');
        this.statusText = document.getElementById('pipeline-status-text');
        this.timerEl = document.getElementById('pipeline-timer');
        this.modeBadge = document.getElementById('expand-mode-badge');
        this.groundedBadge = document.getElementById('expand-grounded-badge');
        this.answerContent = document.getElementById('omni-answer-content');

        this.stages = [];
        this.startTime = null;
        this.timerInterval = null;
        this.isActive = false;
    }

    start(type = 'query', customTitle = null) {
        if (!this.expandCard) return;

        this.isActive = true;
        this.expandCard.classList.remove('hidden');
        this.expandCard.classList.remove('minimized');
        const minBtn = document.getElementById('minimize-expand-card-btn');
        if (minBtn) {
            minBtn.innerHTML = '<i data-lucide="minus"></i>';
            minBtn.title = "Minimize Progress";
        }
        this.answerContent.innerHTML = '';
        this.startTime = Date.now();
        this.startTimer();

        if (type === 'ingest') {
            this.modeBadge.innerText = customTitle || '📦 Multimodal Graph Ingestion';
            this.modeBadge.className = 'route-pill';
            this.groundedBadge.classList.add('hidden');
            this.stages = [
                { id: 'parsing', name: 'Document Parsing' },
                { id: 'chunking', name: 'Horizon Chunking' },
                { id: 'extraction', name: 'Knowledge Extraction' },
                { id: 'resolution', name: 'Alias Disambiguation' },
                { id: 'storage', name: 'LadybugDB Storage' }
            ];
        } else {
            this.modeBadge.innerText = customTitle || '🔍 Multi-Hop Hybrid Retrieval';
            this.modeBadge.className = 'route-pill';
            this.groundedBadge.classList.add('hidden');
            this.stages = [
                { id: 'router', name: 'Concept Extraction' },
                { id: 'retrieval', name: 'Cypher Graph Traversal' },
                { id: 'synthesis', name: 'Contextual Synthesis' },
                { id: 'grader', name: 'Factuality Grader' }
            ];
        }

        this.renderStages();
        this.setStage(0, "Initiating pipeline execution...");
    }

    startTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => {
            if (!this.startTime) return;
            const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
            if (this.timerEl) this.timerEl.innerText = `${elapsed}s`;
        }, 100);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    renderStages() {
        if (!this.stepperStrip) return;
        this.stepperStrip.innerHTML = '';
        this.stages.forEach((stage, idx) => {
            const stepEl = document.createElement('div');
            stepEl.className = 'pipeline-mini-step';
            stepEl.id = `step-node-${idx}`;
            stepEl.innerHTML = `
                <div class="mini-step-node">${idx + 1}</div>
                <span class="mini-step-label">${stage.name}</span>
            `;
            this.stepperStrip.appendChild(stepEl);
        });
        if (window.lucide) lucide.createIcons();
    }

    renderCompletedStages(type = 'query') {
        if (!this.stepperStrip) return;
        this.stopTimer();
        if (type === 'ingest') {
            this.stages = [
                { id: 'parsing', name: 'Document Parsing' },
                { id: 'chunking', name: 'Horizon Chunking' },
                { id: 'extraction', name: 'Knowledge Extraction' },
                { id: 'resolution', name: 'Alias Disambiguation' },
                { id: 'storage', name: 'LadybugDB Storage' }
            ];
        } else {
            this.stages = [
                { id: 'router', name: 'Concept Extraction' },
                { id: 'retrieval', name: 'Cypher Graph Traversal' },
                { id: 'synthesis', name: 'Contextual Synthesis' },
                { id: 'grader', name: 'Factuality Grader' }
            ];
        }
        this.renderStages();
        this.stages.forEach((_, idx) => {
            const stepEl = document.getElementById(`step-node-${idx}`);
            if (stepEl) {
                stepEl.classList.remove('active');
                stepEl.classList.add('completed');
                const node = stepEl.querySelector('.mini-step-node, .step-node');
                if (node) {
                    node.innerHTML = '<i data-lucide="check"></i>';
                }
            }
        });
        if (window.lucide) lucide.createIcons();
    }

    async setStage(stageIdx, statusMessage, minDelay = 60) {
        if (typeof minDelay !== 'number') {
            minDelay = typeof arguments[3] === 'number' ? arguments[3] : 60;
        }
        if (stageIdx < 0 || stageIdx >= this.stages.length) return;

        // Mark previous steps completed with standard green checkmark ticks
        for (let i = 0; i < stageIdx; i++) {
            const prev = document.getElementById(`step-node-${i}`);
            if (prev) {
                prev.classList.remove('active');
                prev.classList.add('completed');
                const node = prev.querySelector('.mini-step-node, .step-node');
                if (node) {
                    node.innerHTML = '<i data-lucide="check"></i>';
                }
            }
        }

        // Mark current step active with spinning loader
        const cur = document.getElementById(`step-node-${stageIdx}`);
        if (cur) {
            cur.classList.add('active');
            cur.classList.remove('completed');
            const node = cur.querySelector('.mini-step-node, .step-node');
            if (node) {
                node.innerHTML = '<i data-lucide="loader-2" class="spin"></i>';
            }
        }

        // Reset upcoming steps
        for (let i = stageIdx + 1; i < this.stages.length; i++) {
            const next = document.getElementById(`step-node-${i}`);
            if (next) {
                next.classList.remove('active');
                next.classList.remove('completed');
                const node = next.querySelector('.mini-step-node, .step-node');
                if (node) {
                    node.innerHTML = `${i + 1}`;
                }
            }
        }

        if (this.statusText) this.statusText.innerText = statusMessage;
        lucide.createIcons();

        if (minDelay > 0) {
            await new Promise(r => setTimeout(r, minDelay));
        }
    }

    async complete(finalMessage = "Pipeline execution complete!") {
        this.isActive = false;
        if (this.expandCard && this.expandCard.classList.contains('minimized')) {
            this.expandCard.classList.remove('minimized');
            const minBtn = document.getElementById('minimize-expand-card-btn');
            if (minBtn) {
                minBtn.innerHTML = '<i data-lucide="minus"></i>';
                minBtn.title = "Minimize Progress";
            }
        }
        this.stages.forEach((_, i) => {
            const el = document.getElementById(`step-node-${i}`);
            if (el) {
                el.classList.remove('active');
                el.classList.add('completed');
                const node = el.querySelector('.mini-step-node, .step-node');
                if (node) {
                    node.innerHTML = '<i data-lucide="check"></i>';
                }
            }
        });
        if (this.statusText) this.statusText.innerText = finalMessage;
        this.stopTimer();
        lucide.createIcons();
    }

    error(errMsg) {
        this.isActive = false;
        if (this.expandCard && this.expandCard.classList.contains('minimized')) {
            this.expandCard.classList.remove('minimized');
        }
        if (this.statusText) this.statusText.innerText = `Error: ${errMsg}`;
        this.stopTimer();
        const activeNode = document.querySelector('.pipeline-mini-step.active, .pipeline-step.active');
        if (activeNode) {
            const node = activeNode.querySelector('.mini-step-node, .step-node');
            if (node) {
                node.innerHTML = '<i data-lucide="alert-triangle"></i>';
                node.style.borderColor = 'var(--accent-red)';
                node.style.color = 'var(--accent-red)';
            }
        }
        lucide.createIcons();
    }
}

const pipelineStepper = new PacedPipelineStepper();

// ==========================================================================
// Raycast / Spotlight Omni-Bar Handlers
// ==========================================================================
function setupOmniBar() {
    const omniModeQuery = document.getElementById('omni-mode-query');
    const omniModeIngest = document.getElementById('omni-mode-ingest');
    const omniMainInput = document.getElementById('omni-main-input');
    const omniSubmitBtn = document.getElementById('omni-submit-btn');
    const omniAttachBtn = document.getElementById('omni-attach-btn');
    const omniFileInput = document.getElementById('omni-file-input');
    const omniFileChip = document.getElementById('omni-file-chip');
    const chipFilename = document.getElementById('chip-filename');
    const chipRemoveBtn = document.getElementById('chip-remove-btn');
    const chipIcon = document.getElementById('chip-icon');
    const omniSamplesBtn = document.getElementById('omni-samples-btn');
    const samplesMenu = document.getElementById('samples-menu');
    const minimizeExpandCardBtn = document.getElementById('minimize-expand-card-btn');
    const closeExpandCardBtn = document.getElementById('close-expand-card-btn');
    const omniExpandCard = document.getElementById('omni-expand-card');

    function toggleMinimizeExpandCard() {
        if (!omniExpandCard) return;
        const isMin = omniExpandCard.classList.contains('minimized');
        if (isMin) {
            omniExpandCard.classList.remove('minimized');
            if (minimizeExpandCardBtn) {
                minimizeExpandCardBtn.innerHTML = '<i data-lucide="minus"></i>';
                minimizeExpandCardBtn.title = "Minimize Progress";
            }
        } else {
            omniExpandCard.classList.add('minimized');
            if (minimizeExpandCardBtn) {
                minimizeExpandCardBtn.innerHTML = '<i data-lucide="maximize-2"></i>';
                minimizeExpandCardBtn.title = "Expand Progress";
            }
        }
        if (window.lucide) lucide.createIcons();
    }

    if (minimizeExpandCardBtn) {
        minimizeExpandCardBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleMinimizeExpandCard();
        });
    }

    if (omniExpandCard) {
        omniExpandCard.addEventListener('click', (e) => {
            if (omniExpandCard.classList.contains('minimized')) {
                toggleMinimizeExpandCard();
            }
        });
    }

    if (closeExpandCardBtn && omniExpandCard) {
        closeExpandCardBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (pipelineStepper.isActive) {
                toggleMinimizeExpandCard();
                showToast("Pipeline running in background. Minimized to bottom bar.", "info");
            } else {
                omniExpandCard.classList.add('hidden');
                omniExpandCard.classList.remove('minimized');
            }
        });
    }

    // Dismiss or minimize omni-expand-card when clicking outside or pressing Escape
    document.addEventListener('pointerdown', (e) => {
        if (!omniExpandCard || omniExpandCard.classList.contains('hidden')) return;

        const omniContainer = document.querySelector('.floating-omni-container');
        const isClickInsideCard = omniExpandCard.contains(e.target);
        const isClickInsideOmni = omniContainer && omniContainer.contains(e.target);
        const isClickSessionCard = e.target.closest && e.target.closest('.session-card-item');

        if (!isClickInsideCard && !isClickInsideOmni && !isClickSessionCard) {
            if (pipelineStepper.isActive) {
                // While active, clicking outside minimizes instead of closing/destroying!
                if (!omniExpandCard.classList.contains('minimized')) {
                    toggleMinimizeExpandCard();
                }
            } else {
                // Once finished, clicking outside closes it cleanly
                omniExpandCard.classList.add('hidden');
                omniExpandCard.classList.remove('minimized');
            }
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && omniExpandCard && !omniExpandCard.classList.contains('hidden')) {
            if (pipelineStepper.isActive) {
                toggleMinimizeExpandCard();
            } else {
                omniExpandCard.classList.add('hidden');
                omniExpandCard.classList.remove('minimized');
            }
        }
    });

    // Mode Toggle
    if (omniModeQuery && omniModeIngest) {
        omniModeQuery.addEventListener('click', () => {
            omniMode = 'query';
            omniModeQuery.classList.add('active');
            omniModeIngest.classList.remove('active');
            omniMainInput.placeholder = "Ask a question...";
        });

        omniModeIngest.addEventListener('click', () => {
            omniMode = 'ingest';
            omniModeIngest.classList.add('active');
            omniModeQuery.classList.remove('active');
            omniMainInput.placeholder = "Type text, drop files, or select a sample...";
        });
    }

    // Ephemeral Single Attachment Chip Renderer
    function renderAttachmentChip() {
        const strip = document.getElementById('omni-attachments-strip');
        if (!strip) return;

        if (!attachedFile) {
            strip.classList.add('hidden');
            strip.innerHTML = '';
            return;
        }

        strip.classList.remove('hidden');
        let iconName = 'file';
        if (attachedFile.name.match(/\.(mp3|wav|m4a|ogg|flac)$/i)) iconName = 'mic';
        else if (attachedFile.type && attachedFile.type.startsWith('image/')) iconName = 'image';
        else if (attachedFile.name.endsWith('.pdf')) iconName = 'file-text';
        else if (attachedFile.name.match(/\.(md|txt|json|csv)$/i)) iconName = 'code-2';

        strip.innerHTML = `
            <div class="omni-attachment-chip">
                <i data-lucide="${iconName}" style="width:13px;height:13px;"></i>
                <span class="chip-name" title="${attachedFile.name}">${attachedFile.name}</span>
                <span class="chip-size">${formatBytes(attachedFile.size)}</span>
                <button class="chip-remove" type="button" id="omni-remove-file-btn" title="Remove file">
                    <i data-lucide="x" style="width:12px;height:12px;"></i>
                </button>
            </div>
        `;

        if (window.lucide) lucide.createIcons();

        const removeBtn = document.getElementById('omni-remove-file-btn');
        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeAttachment();
            });
        }
    }

    function removeAttachment() {
        attachedFile = null;
        renderAttachmentChip();
        if (omniFileInput) omniFileInput.value = '';
        showToast("Removed attachment", "info");
    }
    window.removeAttachment = removeAttachment;

    function handleFileSelected(file) {
        if (!file) return;
        attachedFile = file;
        renderAttachmentChip();
        if (omniModeIngest) omniModeIngest.click();
        showToast(`Attached "${file.name}" (${formatBytes(file.size)})`, "info");
    }
    window.handleGlobalFilesDrop = (fileList) => {
        if (fileList && fileList.length > 0) {
            handleFileSelected(fileList[0]);
        }
    };

    // File Attachment Input Trigger
    if (omniAttachBtn && omniFileInput) {
        omniAttachBtn.addEventListener('click', () => omniFileInput.click());

        omniFileInput.addEventListener('change', () => {
            if (omniFileInput.files && omniFileInput.files.length > 0) {
                handleFileSelected(omniFileInput.files[0]);
            }
            omniFileInput.value = '';
        });
    }

    // Sample Datasets Dropdown
    if (omniSamplesBtn && samplesMenu) {
        omniSamplesBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            samplesMenu.classList.toggle('hidden');
        });

        document.addEventListener('click', (e) => {
            if (!samplesMenu.contains(e.target) && e.target !== omniSamplesBtn) {
                samplesMenu.classList.add('hidden');
            }
        });

        document.querySelectorAll('.sample-item').forEach(item => {
            item.addEventListener('click', () => {
                const sampleKey = item.dataset.sample;
                const sampleText = SAMPLE_DATASETS[sampleKey];
                if (sampleText) {
                    attachedFile = null;
                    renderAttachmentChip();
                    if (omniModeIngest) omniModeIngest.click();
                    omniMainInput.value = sampleText.replace(/\n+/g, ' ').substring(0, 180) + '...';
                    omniMainInput.dataset.fullText = sampleText;
                    samplesMenu.classList.add('hidden');
                    showToast(`Loaded sample dataset: ${sampleKey.toUpperCase()}. Click arrow to ingest!`, "info");
                }
            });
        });
    }

    // Execution Trigger (Enter or Submit Button)
    if (omniMainInput) {
        omniMainInput.addEventListener('input', () => {
            delete omniMainInput.dataset.fullText;
        });

        omniMainInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                executeOmniAction();
            }
        });
    }

    if (omniSubmitBtn) {
        omniSubmitBtn.addEventListener('click', executeOmniAction);
    }

    async function executeOmniAction() {
        const text = (omniMainInput.dataset.fullText || omniMainInput.value || '').trim();

        // Determine if this is an Ingest or Query action
        const isIngest = (omniMode === 'ingest') || (attachedFile !== null) || (text.length > 250);

        if (isIngest) {
            await handleIngestExecution(text);
        } else {
            await handleQueryExecution(text);
        }
    }

    async function handleIngestExecution(text) {
        if (!attachedFile && !text) {
            showToast("Please enter text or attach a file to ingest.", "error");
            return;
        }

        omniSubmitBtn.disabled = true;
        const ingestLabel = attachedFile 
            ? `Ingesting: ${attachedFile.name}` 
            : 'Ingesting Knowledge Payload';
        pipelineStepper.start('ingest', ingestLabel);

        try {
            const maxChunkTokens = parseInt(localStorage.getItem('graphrag_max_chunk_size') || '800', 10);
            let response;
            if (attachedFile) {
                const formData = new FormData();
                formData.append('file', attachedFile);
                formData.append('clear_db', 'false');
                formData.append('max_chunk_tokens', maxChunkTokens.toString());

                await pipelineStepper.setStage(0, `Uploading and analyzing ${attachedFile.name}...`, 60);
                response = await fetchApi('/ingest_file', {
                    method: 'POST',
                    body: formData
                });
            } else {
                await pipelineStepper.setStage(0, `Analyzing raw text payload (${text.length.toLocaleString()} characters)...`, 60);
                response = await fetchApi('/ingest', {
                    method: 'POST',
                    body: JSON.stringify({ 
                        text: text, 
                        clear_db: false,
                        max_chunk_tokens: maxChunkTokens 
                    })
                });
            }

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Ingestion request failed");
            }

            // Stream SSE progress
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const event = JSON.parse(line.substring(6));

                        if (event.type === 'progress') {
                            const stageName = (event.stage || '').toLowerCase();
                            if (stageName.includes('pars') || stageName.includes('vision') || stageName.includes('audio')) {
                                await pipelineStepper.setStage(0, event.status, 40);
                            } else if (stageName.includes('chunk')) {
                                await pipelineStepper.setStage(1, event.status, 40);
                            } else if (stageName.includes('extract')) {
                                await pipelineStepper.setStage(2, event.status, 40);
                            } else if (stageName.includes('resol')) {
                                await pipelineStepper.setStage(3, event.status, 40);
                            } else if (stageName.includes('communit') || stageName.includes('storage')) {
                                await pipelineStepper.setStage(4, event.status, 40);
                            }
                        } else if (event.type === 'result') {
                            const res = event.results;
                            latestChunks = res.chunks || [];
                            latestAliases = res.alias_resolutions || [];

                            await pipelineStepper.complete(`Successfully indexed ${res.entities_extracted} entities and ${res.relations_extracted} relationships into LadybugDB!`);

                            // Render summary card in answer content
                            const answerEl = document.getElementById('omni-answer-content');
                            if (answerEl) {
                                answerEl.innerHTML = `
                                    <div class="ingest-summary-card">
                                        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
                                            <i data-lucide="check-circle" style="color:var(--accent-green);width:20px;height:20px;"></i>
                                            <h4 style="margin:0;color:#fff;font-size:1.05rem;">Knowledge Ingestion Complete</h4>
                                        </div>
                                        <p style="color:var(--text-secondary);font-size:0.875rem;margin-bottom:1rem;">
                                            The document was situatively chunked, processed by Gemini 3.8-Flash, and persisted as property nodes and edges in LadybugDB.
                                        </p>
                                        <div class="stats-badge-strip" style="justify-content:flex-start;">
                                            <div class="stat-badge"><strong>${res.entities_extracted}</strong> <span>New Entities</span></div>
                                            <div class="stat-sep">•</div>
                                            <div class="stat-badge"><strong>${res.relations_extracted}</strong> <span>Relationships</span></div>
                                            <div class="stat-sep">•</div>
                                            <div class="stat-badge"><strong>${res.chunks_created}</strong> <span>Contextual Chunks</span></div>
                                        </div>
                                    </div>
                                `;
                                lucide.createIcons();
                            }

                            showToast(`Graph updated: +${res.entities_extracted} entities, +${res.relations_extracted} relations!`, "success");

                            // Persist to session history
                            saveSessionRecord({
                                id: 'sess_' + Date.now(),
                                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                                type: 'ingest',
                                title: `Ingested ${res.chunks_created || latestChunks.length} chunks & ${res.entities_extracted || latestAliases.length} concepts`,
                                snippet: `Indexed ${res.entities_extracted || 0} entities and ${res.relations_extracted || 0} relationships into LadybugDB.`,
                                answer: `### Knowledge Ingestion Summary\n- **Entities Indexed:** ${res.entities_extracted || 0}\n- **Relationships Stored:** ${res.relations_extracted || 0}\n- **Contextual Chunks:** ${res.chunks_created || 0}\n\n*The text was situatively enriched and persisted into LadybugDB.*`,
                                chunks: latestChunks,
                                aliases: latestAliases,
                                trace: null
                            });

                            // Reset input & attachments
                            attachedFile = null;
                            renderAttachmentChip();
                            if (omniFileInput) omniFileInput.value = '';
                            omniMainInput.value = '';
                            delete omniMainInput.dataset.fullText;

                            // Reload graph and counters
                            loadGraphData();
                            refreshStats();
                        } else if (event.type === 'error') {
                            throw new Error(event.detail || "Pipeline failed");
                        }
                    }
                }
            }
        } catch (e) {
            pipelineStepper.error(e.message);
            showToast(`Ingestion error: ${e.message}`, "error");
        } finally {
            omniSubmitBtn.disabled = false;
        }
    }

    async function handleQueryExecution(query) {
        if (!query) {
            showToast("Please enter a question to ask the knowledge graph.", "info");
            return;
        }

        omniSubmitBtn.disabled = true;
        pipelineStepper.start('query', `Querying: "${query.substring(0, 35)}..."`);

        try {
            await pipelineStepper.setStage(0, `Extracting query concepts & entity anchors...`, 80);

            // Execute LangGraph query
            const resPromise = fetchApi('/query', {
                method: 'POST',
                body: JSON.stringify({ query: query, query_type: 'auto' })
            });

            await pipelineStepper.setStage(1, `Traversing Cypher property graph paths & neighborhoods...`, 80);
            await pipelineStepper.setStage(2, `Synthesizing contextual multi-hop answer with Gemini 3.8-Flash...`, 80);

            const res = await resPromise;
            const data = await res.json();
            if (!res.ok || data.status !== 'success') {
                throw new Error(data.detail || "Query failed");
            }

            const results = data.results;
            latestQueryTrace = {
                query,
                route: results.route || 'hybrid',
                extracted_entities: results.extracted_entities || [],
                graph_context: results.graph_context || '',
                vector_context: results.vector_context || '',
                is_grounded: results.is_grounded !== false,
                retry_count: results.retry_count || 0
            };

            await pipelineStepper.setStage(3, `Verifying factual grounding against retrieved knowledge triples...`, 80);
            await pipelineStepper.complete("Grounded response synthesized across knowledge graph and passages.");

            // Update Header Badges in reasoning card
            const modeBadge = document.getElementById('expand-mode-badge');
            const groundedBadge = document.getElementById('expand-grounded-badge');
            if (modeBadge) {
                const routeLabel = results.route === 'graph' ? '🧭 Cypher Graph Traversal' : (results.route === 'vector' ? '📄 Contextual Text Vector' : '🔍 Multi-Hop Hybrid Retrieval');
                modeBadge.innerText = routeLabel;
            }
            if (groundedBadge) {
                groundedBadge.classList.remove('hidden');
                if (results.is_grounded !== false) {
                    groundedBadge.innerHTML = '<i data-lucide="shield-check"></i> Grounded';
                    groundedBadge.style.color = 'var(--accent-green)';
                    groundedBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                } else {
                    groundedBadge.innerHTML = '<i data-lucide="alert-triangle"></i> Partial';
                    groundedBadge.style.color = 'var(--accent-yellow)';
                    groundedBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                }
            }

            // Render Markdown Answer
            const answerEl = document.getElementById('omni-answer-content');
            if (answerEl) {
                const answerHtml = window.marked ? marked.parse(results.answer || "No response generated.") : results.answer;
                let triplesHtml = '';

                if (results.relations && results.relations.length > 0) {
                    triplesHtml = `
                        <div class="retrieved-triples-section" style="margin-top:1rem;border-top:1px solid var(--border-glass);padding-top:0.75rem;">
                            <span class="sec-label">Retrieved Graph Triples (${results.relations.length}):</span>
                            <div class="triples-list" style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem;">
                                ${results.relations.slice(0, 6).map(r => `
                                    <div class="triple-chip" style="cursor:pointer;" onclick="focusNode('${r.source_id || r.source_name}')">
                                        <span class="triple-sub">${r.source_name || r.source_id}</span>
                                        <span class="triple-rel">${r.type}</span>
                                        <span class="triple-obj">${r.target_name || r.target_id}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                answerEl.innerHTML = `
                    <div class="query-answer-body">
                        ${answerHtml}
                        ${triplesHtml}
                    </div>
                `;
                lucide.createIcons();
            }

            // Highlight queried entities in graph
            if (results.entities && results.entities.length > 0) {
                highlightQueryEntities(results.entities);
            }

            // Persist to session history
            saveSessionRecord({
                id: 'sess_' + Date.now(),
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                type: 'query',
                title: query,
                snippet: results.answer ? (results.answer.length > 90 ? results.answer.substring(0, 90) + '...' : results.answer) : 'Query executed.',
                answer: results.answer,
                trace: latestQueryTrace,
                chunks: latestChunks,
                aliases: latestAliases
            });
        } catch (e) {
            pipelineStepper.error(e.message);
            const answerEl = document.getElementById('omni-answer-content');
            if (answerEl) {
                answerEl.innerHTML = `<div class="empty-state text-danger">${e.message}</div>`;
            }
        } finally {
            omniSubmitBtn.disabled = false;
        }
    }
}

// Full Window Drag and Drop Overlay
function setupFullWindowDrop() {
    const dropOverlay = document.getElementById('window-drop-overlay');
    if (!dropOverlay) return;

    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        dropOverlay.classList.remove('hidden');
    });

    window.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dropOverlay.classList.add('hidden');
        }
    });

    window.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    window.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        dropOverlay.classList.add('hidden');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            if (typeof window.handleGlobalFilesDrop === 'function') {
                window.handleGlobalFilesDrop(e.dataTransfer.files);
            }
        }
    });
}

// ==========================================================================
// Vis.js Interactive Knowledge Graph
// ==========================================================================
function isNodeVisible(node) {
    if (activeFilter.selectedCommunities && activeFilter.selectedCommunities.size > 0) {
        if (!activeFilter.selectedCommunities.has(String(node.community_id))) {
            return false;
        }
    }
    if (activeFilter.entityType !== 'all' && (node.type || '').toUpperCase() !== activeFilter.entityType.toUpperCase()) {
        return false;
    }
    if (activeFilter.selectedNodeId && activeFilter.hopDepth !== 'all') {
        if (!activeNeighborhoodIds || !activeNeighborhoodIds.has(node.id)) {
            return false;
        }
    }
    return true;
}

function isEdgeVisible(edge) {
    const src = nodesDS ? nodesDS.get(edge.from) : null;
    const tgt = nodesDS ? nodesDS.get(edge.to) : null;
    if (!src || !tgt) return false;
    return isNodeVisible(src) && isNodeVisible(tgt);
}

function getRawConnectedNodes(nodeId) {
    if (!edgesDS) return [];
    const connected = [];
    const allEdges = edgesDS.get();
    for (let i = 0; i < allEdges.length; i++) {
        const edge = allEdges[i];
        if (edge.from === nodeId) connected.push(edge.to);
        else if (edge.to === nodeId) connected.push(edge.from);
    }
    return connected;
}

function recomputeNeighborhood() {
    if (!activeFilter.selectedNodeId || activeFilter.hopDepth === 'all') {
        activeNeighborhoodIds = null;
        return;
    }
    const maxDepth = parseInt(activeFilter.hopDepth, 10);
    const visited = new Set([activeFilter.selectedNodeId]);
    let currentLevel = new Set([activeFilter.selectedNodeId]);

    for (let d = 0; d < maxDepth; d++) {
        const nextLevel = new Set();
        currentLevel.forEach(nId => {
            const neighbors = getRawConnectedNodes(nId);
            neighbors.forEach(nb => {
                if (!visited.has(nb)) {
                    visited.add(nb);
                    nextLevel.add(nb);
                }
            });
        });
        currentLevel = nextLevel;
    }
    activeNeighborhoodIds = visited;
}

function refreshGraphFilters() {
    recomputeNeighborhood();
    if (nodesView) nodesView.refresh();
    if (edgesView) edgesView.refresh();
}

function initGraph() {
    if (network) return;
    const container = document.getElementById('network-container');
    if (!container) return;

    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);
    nodesView = new vis.DataView(nodesDS, { filter: isNodeVisible });
    edgesView = new vis.DataView(edgesDS, { filter: isEdgeVisible });

    const data = { nodes: nodesView, edges: edgesView };

    const options = {
        nodes: {
            shape: 'dot',
            size: 22,
            font: {
                size: 13,
                color: '#f8fafc',
                face: 'Outfit',
                strokeWidth: 4,
                strokeColor: '#05070d'
            },
            borderWidth: 2,
            shadow: {
                enabled: true,
                color: 'rgba(0, 0, 0, 0.65)',
                size: 12,
                x: 2, y: 3
            }
        },
        edges: {
            width: 1.5,
            color: { color: 'rgba(255, 255, 255, 0.22)', highlight: '#06b6d4', hover: '#818cf8' },
            smooth: { type: 'curvedCW', roundness: 0.18 },
            arrows: { to: { enabled: true, scaleFactor: 0.55 } },
            font: {
                size: 10,
                color: '#94a3b8',
                face: 'Outfit',
                strokeWidth: 3,
                strokeColor: '#05070d',
                align: 'middle'
            }
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            stabilization: {
                enabled: true,
                iterations: 350,
                updateInterval: 25,
                fit: true
            },
            forceAtlas2Based: {
                gravitationalConstant: -75,
                centralGravity: 0.008,
                springLength: 175,
                springConstant: 0.07,
                damping: 0.4,
                avoidOverlap: 1.0
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 150,
            navigationButtons: false,
            keyboard: false
        }
    };

    network = new vis.Network(container, data, options);

    // Node click inspector listener
    network.on('click', (params) => {
        if (params.nodes && params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            showNodeInspector(nodeId);
        } else {
            closeAllFloatingPanels();
            closeNodeInspector();
        }
    });
}

async function loadGraphData() {
    try {
        const res = await fetchApi('/graph');
        const data = await res.json();
        if (data.status === 'success' && data.results) {
            updateGraphCanvas(data.results);
        }
    } catch (e) {
        console.error("Failed to load graph:", e);
    }
}

function updateGraphCanvas(results) {
    const { entities, relations } = results;
    if (!nodesDS || !edgesDS) return;

    const existingNodeIds = new Set(nodesDS.getIds());
    const existingEdgeIds = new Set(edgesDS.getIds());

    const communitySet = new Set();
    const typeSet = new Set();

    // Compute informative cluster names from top connected entities
    const commEntities = {};
    const entDegrees = {};
    entities.forEach(ent => {
        const commId = ent.community_id !== undefined && ent.community_id !== null && ent.community_id !== -1 ? ent.community_id : 0;
        if (!commEntities[commId]) commEntities[commId] = [];
        commEntities[commId].push(ent);
        entDegrees[ent.id] = 0;
    });

    relations.forEach(rel => {
        if (entDegrees[rel.source_id] !== undefined) entDegrees[rel.source_id]++;
        if (entDegrees[rel.target_id] !== undefined) entDegrees[rel.target_id]++;
    });

    const clusterNames = {};
    Object.keys(commEntities).forEach(cid => {
        const ents = commEntities[cid];
        ents.sort((a, b) => (entDegrees[b.id] || 0) - (entDegrees[a.id] || 0));
        const topNames = ents.slice(0, 2).map(e => e.name || e.id).filter(Boolean);
        const title = topNames.length > 0 ? topNames.join(' & ') : `Cluster #${cid}`;
        clusterNames[cid] = `${title} (#${cid})`;
    });
    window.CLUSTER_NAMES = clusterNames;

    entities.forEach(ent => {
        const commId = ent.community_id !== undefined && ent.community_id !== null && ent.community_id !== -1 ? ent.community_id : 0;
        communitySet.add(commId);
        if (ent.type) typeSet.add(ent.type);
        const nodeColor = COMMUNITY_COLORS[commId % COMMUNITY_COLORS.length];

        const nodeObj = {
            id: ent.id,
            label: ent.name || ent.id,
            type: ent.type || 'Concept',
            community_id: commId,
            description: ent.description || '',
            color: {
                background: nodeColor,
                border: '#ffffff',
                highlight: { background: nodeColor, border: '#06b6d4' }
            }
        };

        if (existingNodeIds.has(ent.id)) {
            nodesDS.update(nodeObj);
        } else {
            nodesDS.add(nodeObj);
        }
    });

    relations.forEach(rel => {
        const edgeId = `${rel.source_id}-${rel.type}-${rel.target_id}`;
        const edgeObj = {
            id: edgeId,
            from: rel.source_id,
            to: rel.target_id,
            label: rel.type,
            title: rel.description || rel.type
        };

        if (existingEdgeIds.has(edgeId)) {
            edgesDS.update(edgeObj);
        } else {
            edgesDS.add(edgeObj);
        }
    });

    renderCommunityLegend(communitySet);
    populateFilterDropdowns(communitySet, typeSet);
}

function populateFilterDropdowns(communitySet, typeSet) {
    const typeSelect = document.getElementById('filter-type');
    if (typeSelect) {
        const curVal = typeSelect.value;
        typeSelect.innerHTML = '<option value="all">All Entity Types</option>';
        Array.from(typeSet).sort().forEach(t => {
            typeSelect.innerHTML += `<option value="${t}">${t}</option>`;
        });
        typeSelect.value = curVal || 'all';
    }
}

function renderCommunityLegend(communitySet) {
    window.CURRENT_COMMUNITY_SET = communitySet;
    const container = document.getElementById('community-legend-chips');
    if (!container) return;
    container.innerHTML = '';

    if (!communitySet || communitySet.size === 0) {
        container.innerHTML = '<span class="legend-item">None</span>';
        return;
    }

    // Interactive "All" toggle button (active when no specific clusters are filtered)
    const isAllActive = !activeFilter.selectedCommunities || activeFilter.selectedCommunities.size === 0;
    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = `cluster-chip-btn ${isAllActive ? 'active' : ''}`;
    allBtn.innerText = 'All';
    allBtn.title = 'Show all clusters';
    allBtn.addEventListener('click', () => {
        activeFilter.selectedCommunities.clear();
        refreshGraphFilters();
        renderCommunityLegend(communitySet);
    });
    container.appendChild(allBtn);

    // Individual interactive cluster chips
    Array.from(communitySet).sort((a, b) => a - b).forEach(cid => {
        const color = COMMUNITY_COLORS[cid % COMMUNITY_COLORS.length];
        const label = (window.CLUSTER_NAMES && window.CLUSTER_NAMES[cid]) || `Cluster #${cid}`;
        const cidStr = String(cid);
        const isClusterActive = activeFilter.selectedCommunities && activeFilter.selectedCommunities.has(cidStr);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `cluster-chip-btn ${isClusterActive ? 'active' : ''}`;
        btn.title = `Filter by ${label}`;
        btn.innerHTML = `<span class="cluster-chip-dot" style="background: ${color};"></span><span>${label}</span>`;

        btn.addEventListener('click', () => {
            if (activeFilter.selectedCommunities.has(cidStr)) {
                activeFilter.selectedCommunities.delete(cidStr);
            } else {
                activeFilter.selectedCommunities.add(cidStr);
            }
            refreshGraphFilters();
            renderCommunityLegend(communitySet);
        });

        container.appendChild(btn);
    });
}

// Slide-Over Entity Inspector Drawer (Memgraph Lab / Neo4j Bloom)
function showNodeInspector(nodeId) {
    closeAllFloatingPanels('bloom-inspector');
    const node = nodesDS.get(nodeId);
    if (!node) return;

    const drawer = document.getElementById('bloom-inspector');
    if (!drawer) return;

    const cid = node.community_id ?? 0;
    const commName = (window.CLUSTER_NAMES && window.CLUSTER_NAMES[cid]) || `Cluster #${cid}`;

    document.getElementById('insp-name').innerText = node.label || node.id;
    document.getElementById('insp-type').innerText = node.type || 'Entity';
    document.getElementById('insp-comm').innerText = commName;
    document.getElementById('insp-desc').innerText = node.description || "No synthesized summary available.";

    // Connected Neighbors
    const connectedNodeIds = network.getConnectedNodes(nodeId);
    const neighborsList = document.getElementById('insp-neighbors');
    neighborsList.innerHTML = '';

    if (connectedNodeIds.length === 0) {
        neighborsList.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">No connected relations.</span>';
    } else {
        connectedNodeIds.forEach(nId => {
            const neighbor = nodesDS.get(nId);
            if (neighbor) {
                const chip = document.createElement('div');
                chip.className = 'neighbor-chip';
                chip.innerHTML = `<span>${neighbor.label || neighbor.id}</span><i data-lucide="arrow-up-right" style="width:14px;height:14px;"></i>`;
                chip.addEventListener('click', () => {
                    focusNode(nId);
                    showNodeInspector(nId);
                });
                neighborsList.appendChild(chip);
            }
        });
    }

    // Isolate 1-Hop & 2-Hop Buttons
    const iso1Btn = document.getElementById('insp-isolate-1hop-btn');
    if (iso1Btn) {
        iso1Btn.onclick = () => {
            activeFilter.selectedNodeId = nodeId;
            activeFilter.hopDepth = '1';
            const hopsSelect = document.getElementById('filter-hops');
            if (hopsSelect) hopsSelect.value = '1';
            refreshGraphFilters();
            network.fit({ animation: { duration: 500 } });
            showToast(`Isolated 1-hop neighborhood for "${node.label || node.id}"`, "info");
        };
    }

    const iso2Btn = document.getElementById('insp-isolate-2hop-btn');
    if (iso2Btn) {
        iso2Btn.onclick = () => {
            activeFilter.selectedNodeId = nodeId;
            activeFilter.hopDepth = '2';
            const hopsSelect = document.getElementById('filter-hops');
            if (hopsSelect) hopsSelect.value = '2';
            refreshGraphFilters();
            network.fit({ animation: { duration: 500 } });
            showToast(`Isolated 2-hop sub-graph for "${node.label || node.id}"`, "info");
        };
    }

    lucide.createIcons();
    drawer.classList.remove('hidden');
}

function closeNodeInspector() {
    const drawer = document.getElementById('bloom-inspector');
    if (drawer) drawer.classList.add('hidden');
}

function focusNode(nodeId) {
    if (!network || !nodesDS.get(nodeId)) return;
    network.focus(nodeId, {
        scale: 1.25,
        animation: { duration: 600, easingFunction: 'easeInOutQuad' }
    });
}

function highlightQueryEntities(entities) {
    if (!network || !entities) return;
    const highlightIds = entities.map(e => e.id || e);
    network.selectNodes(highlightIds);
}

// ==========================================================================
// Canvas Navigation HUD
// ==========================================================================
function setupCanvasHUD() {
    const fitBtn = document.getElementById('fit-graph-btn');
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const stabilizeBtn = document.getElementById('stabilize-btn');
    const physicsToggleBtn = document.getElementById('physics-toggle-btn');
    let physicsEnabled = true;

    if (fitBtn) {
        fitBtn.addEventListener('click', () => {
            if (network) network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        });
    }

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => {
            if (network) {
                const scale = network.getScale();
                network.moveTo({ scale: scale * 1.3, animation: { duration: 300 } });
            }
        });
    }

    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => {
            if (network) {
                const scale = network.getScale();
                network.moveTo({ scale: scale / 1.3, animation: { duration: 300 } });
            }
        });
    }

    if (stabilizeBtn) {
        stabilizeBtn.addEventListener('click', () => {
            if (network) network.stabilize();
        });
    }

    if (physicsToggleBtn) {
        physicsToggleBtn.addEventListener('click', () => {
            if (!network) return;
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled } });
            physicsToggleBtn.style.color = physicsEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)';
            showToast(`Graph physics ${physicsEnabled ? 'enabled' : 'frozen'}.`, "info");
        });
    }

    const closeInspBtn = document.getElementById('close-inspector-btn');
    if (closeInspBtn) {
        closeInspBtn.addEventListener('click', closeNodeInspector);
    }
}

// ==========================================================================
// Sub-Graph Filter Dock Handlers
// ==========================================================================
function setupFilterDock() {
    const typeSelect = document.getElementById('filter-type');
    const resetFiltersBtn = document.getElementById('reset-filters-btn');

    if (typeSelect) {
        typeSelect.addEventListener('change', (e) => {
            activeFilter.entityType = e.target.value;
            refreshGraphFilters();
        });
    }

    const maxChunkSlider = document.getElementById('max-chunk-slider');
    const maxChunkVal = document.getElementById('max-chunk-val');
    if (maxChunkSlider && maxChunkVal) {
        const savedVal = localStorage.getItem('graphrag_max_chunk_size') || '800';
        maxChunkSlider.value = savedVal;
        maxChunkVal.innerText = `${savedVal} tok`;

        maxChunkSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            maxChunkVal.innerText = `${val} tok`;
            localStorage.setItem('graphrag_max_chunk_size', val);
        });
    }

    if (resetFiltersBtn) {
        resetFiltersBtn.addEventListener('click', () => {
            activeFilter = { community: 'all', entityType: 'all', selectedNodeId: null, hopDepth: 'all', selectedCommunities: new Set() };
            activeNeighborhoodIds = null;
            if (typeSelect) typeSelect.value = 'all';
            renderCommunityLegend(window.CURRENT_COMMUNITY_SET || new Set());
            refreshGraphFilters();
            if (network) network.fit({ animation: { duration: 500 } });
            showToast("Reset all filters to global graph.", "info");
        });
    }
}

function setupInspectorDrawer() {
    // Drawer buttons handled in showNodeInspector
}

// ==========================================================================
// Technical Deep Dive Inspection Modal
// ==========================================================================
function setupDeepInspectionModal() {
    const modal = document.getElementById('deep-inspection-modal');
    const openBtn = document.getElementById('inspect-deck-btn');
    const closeBtn = document.getElementById('close-inspection-modal-btn');
    const tabBtns = document.querySelectorAll('.modal-tab-btn');
    const panes = document.querySelectorAll('.modal-pane');

    if (openBtn && modal) {
        openBtn.addEventListener('click', () => {
            closeAllFloatingPanels();
            modal.classList.remove('hidden');
            renderInspectionTabs();
        });
    }

    if (closeBtn && modal) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    }

    if (modal) {
        modal.addEventListener('pointerdown', (e) => {
            if (!e.target.closest('.inspection-modal')) {
                modal.classList.add('hidden');
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    });

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.dataset.modalTab;
            const targetPane = document.getElementById(`modal-pane-${target}`);
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

async function renderInspectionTabs() {
    // 1. Fetch latest chunks if empty
    if (latestChunks.length === 0) {
        try {
            const res = await fetchApi('/chunks');
            const data = await res.json();
            if (data.status === 'success' && data.chunks) {
                latestChunks = data.chunks;
            }
        } catch (e) { }
    }
    renderChunksGrid(latestChunks);

    // 2. Fetch latest aliases if empty
    if (latestAliases.length === 0) {
        try {
            const res = await fetchApi('/aliases');
            const data = await res.json();
            if (data.status === 'success' && data.aliases) {
                latestAliases = data.aliases;
            }
        } catch (e) { }
    }
    renderAliasesTable(latestAliases);

    // 3. Render LangGraph Trace
    renderLangGraphTrace(latestQueryTrace);
}

function renderChunksGrid(chunks) {
    const grid = document.getElementById('chunks-grid');
    if (!grid) return;
    grid.innerHTML = '';

    if (!chunks || chunks.length === 0) {
        grid.innerHTML = '<div class="empty-state">No contextual chunks stored. Ingest a document or sample dataset to view macro-context horizon chunks.</div>';
        return;
    }

    chunks.forEach(chunk => {
        const card = document.createElement('div');
        card.className = 'chunk-card';
        card.innerHTML = `
            <div class="chunk-card-header">
                <span class="chunk-id-badge">${chunk.id}</span>
                <span class="text-muted" style="font-size:0.7rem;font-family:var(--font-mono);">${chunk.text.length} chars</span>
            </div>
            <div class="contextual-prefix-box">
                <span class="prefix-tag">Macro-Context Horizon (Target 400-600 tokens)</span>
                <p>${chunk.context || 'Macro-context horizon bounded.'}</p>
            </div>
            <div class="raw-chunk-text">
                ${chunk.text}
            </div>
        `;
        grid.appendChild(card);
    });
}

function renderAliasesTable(aliases) {
    const tbody = document.getElementById('aliases-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!aliases || aliases.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center empty-state">No alias or disambiguation records available in current session.</td></tr>';
        return;
    }

    aliases.forEach(record => {
        const row = document.createElement('tr');
        const badgeClass = record.resolution_type === 'merged' ? 'merged' : (record.resolution_type === 'homonym' ? 'homonym' : 'new');
        const badgeLabel = record.resolution_type === 'merged' ? 'Merged Concept' : (record.resolution_type === 'homonym' ? 'Homonym Disambiguated' : 'New Concept');

        row.innerHTML = `
            <td style="font-weight:700;color:#fff;">${record.canonical_name}</td>
            <td style="font-family:var(--font-mono);color:var(--text-secondary);">${(record.aliases || []).join(', ') || record.canonical_name}</td>
            <td><span class="entity-type-badge">${record.entity_type || 'Entity'}</span></td>
            <td><span class="resolution-badge ${badgeClass}">${badgeLabel}</span></td>
            <td style="font-size:0.775rem;line-height:1.4;color:var(--text-secondary);">${record.description}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderLangGraphTrace(trace) {
    const container = document.getElementById('langgraph-trace-content');
    if (!container) return;

    if (!trace) {
        container.innerHTML = '<div class="empty-state">No queries run yet. Ask a question using the omni-bar to inspect the full LangGraph agent trace.</div>';
        return;
    }

    container.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:1rem;">
            <div class="glass-card" style="padding:1.25rem;">
                <span class="sec-label">User Query:</span>
                <h3 style="font-size:1rem;color:#fff;margin:0.25rem 0 0.75rem;">"${trace.query}"</h3>
                <div style="display:flex;gap:0.75rem;flex-wrap:wrap;align-items:center;">
                    <span class="route-pill">Architecture: <strong>Deterministic Hybrid</strong></span>
                    <span class="grounding-badge">Factual Grounding: <strong>${trace.is_grounded ? 'VERIFIED (100%)' : 'PARTIAL / RETRY'}</strong></span>
                </div>
                <div style="margin-top:0.75rem;font-size:0.775rem;color:var(--text-secondary);line-height:1.45;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.6rem;">
                    <strong>Pipeline Execution Flow:</strong> 
                    Concept Extraction ➔ Cypher Graph Traversal ➔ Passage & Entity Retrieval ➔ Contextual Synthesis ➔ Factuality Verification Grader.
                    <br>
                    <span style="color:var(--text-muted);font-size:0.75rem;">Deterministic hybrid retrieval fuses multi-hop topological graph paths with raw text passages to prevent hallucination without routing divergence.</span>
                </div>
            </div>

            <div class="control-card">
                <span class="sec-label">Extracted Query Concepts & Entities:</span>
                <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.35rem;">
                    ${(trace.extracted_entities && trace.extracted_entities.length > 0)
                        ? trace.extracted_entities.map(e => `<span class="format-badge">${e}</span>`).join('')
                        : '<span style="font-size:0.775rem;color:var(--text-muted);">None explicitly extracted (Global graph fallback active)</span>'}
                </div>
            </div>

            <div class="control-card">
                <span class="sec-label">Retrieved Cypher Graph Context:</span>
                <pre style="font-family:var(--font-mono);font-size:0.75rem;color:#93c5fd;white-space:pre-wrap;background:rgba(0,0,0,0.3);padding:0.75rem;border-radius:6px;max-height:160px;overflow-y:auto;margin-top:0.35rem;">${trace.graph_context || 'No graph triples fetched for this query.'}</pre>
            </div>

            ${trace.vector_context ? `
            <div class="control-card">
                <span class="sec-label">Retrieved Document Passages & Contextual Excerpts:</span>
                <pre style="font-family:var(--font-mono);font-size:0.75rem;color:#cbd5e1;white-space:pre-wrap;background:rgba(0,0,0,0.3);padding:0.75rem;border-radius:6px;max-height:140px;overflow-y:auto;margin-top:0.35rem;">${trace.vector_context}</pre>
            </div>
            ` : ''}
        </div>
    `;
    if (window.lucide) lucide.createIcons();
}

// ==========================================================================
// LadybugDB Interactive Cypher Console
// ==========================================================================
function setupCypherConsole() {
    const input = document.getElementById('cypher-query-input');
    const runBtn = document.getElementById('run-cypher-btn');
    const resultsContainer = document.getElementById('cypher-results-container');

    if (!runBtn || !input || !resultsContainer) return;

    runBtn.addEventListener('click', executeCypher);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') executeCypher();
    });

    document.querySelectorAll('.cypher-preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            input.value = btn.dataset.cypher;
            executeCypher();
        });
    });

    async function executeCypher() {
        const query = input.value.trim();
        if (!query) return;

        runBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="empty-state">Running Cypher query against LadybugDB...</div>';

        try {
            const res = await fetchApi('/cypher', {
                method: 'POST',
                body: JSON.stringify({ query: query, limit: 50 })
            });
            const data = await res.json();

            if (data.status !== 'success') {
                resultsContainer.innerHTML = `<div class="empty-state text-danger">Cypher Error: ${data.error || 'Syntax error in Cypher query'}</div>`;
                return;
            }

            if (!data.rows || data.rows.length === 0) {
                resultsContainer.innerHTML = '<div class="empty-state">Query executed successfully. 0 rows returned.</div>';
                return;
            }

            // Render table
            let tableHtml = `<table class="modern-table"><thead><tr>`;
            data.columns.forEach(col => {
                tableHtml += `<th>${col}</th>`;
            });
            tableHtml += `</tr></thead><tbody>`;

            data.rows.forEach(row => {
                tableHtml += `<tr>`;
                row.forEach(cell => {
                    tableHtml += `<td>${cell}</td>`;
                });
                tableHtml += `</tr>`;
            });

            tableHtml += `</tbody></table>`;
            resultsContainer.innerHTML = tableHtml;
        } catch (e) {
            resultsContainer.innerHTML = `<div class="empty-state text-danger">${e.message}</div>`;
        } finally {
            runBtn.disabled = false;
        }
    }
}

// ==========================================================================
// Stats & Utilities
// ==========================================================================
async function refreshStats() {
    try {
        const res = await fetchApi('/stats');
        const data = await res.json();
        if (data.status === 'success' && data.stats) {
            const nodesEl = document.getElementById('metric-nodes');
            const relsEl = document.getElementById('metric-relations');
            const chunksEl = document.getElementById('metric-chunks');
            const commsEl = document.getElementById('metric-communities');

            if (nodesEl) nodesEl.innerText = data.stats.nodes ?? 0;
            if (relsEl) relsEl.innerText = data.stats.relations ?? 0;
            if (chunksEl) chunksEl.innerText = data.stats.chunks ?? 0;
            if (commsEl) commsEl.innerText = data.stats.communities ?? 0;
        }
    } catch (e) { }
}

function showToast(message, type = "info") {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const iconMap = {
        success: 'check-circle',
        error: 'alert-circle',
        info: 'info'
    };

    toast.innerHTML = `
        <i data-lucide="${iconMap[type] || 'info'}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);
    lucide.createIcons();

    toast.addEventListener('click', () => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 200);
    });

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 400);
    }, 4500);
}

function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
