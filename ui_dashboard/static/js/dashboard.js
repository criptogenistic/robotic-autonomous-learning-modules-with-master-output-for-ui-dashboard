// Dashboard JavaScript

let selectedDMP = null;
let trajectoryChart = null;
let autoRefreshInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    loadDMPInstances();
    initializeChart();
    startAutoRefresh();
});

function initializeDashboard() {
    console.log('Dashboard initialized');
    document.getElementById('createDMPForm').addEventListener('submit', handleCreateDMP);
    updateSystemStatus();
}

function initializeChart() {
    const ctx = document.getElementById('trajectoryChart').getContext('2d');
    trajectoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Trajectory Y',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Velocity dY',
                    data: [],
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'DMP Trajectory Output'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function startAutoRefresh() {
    autoRefreshInterval = setInterval(() => {
        loadDMPInstances();
        updateSystemStatus();
        loadTrajectory();
        loadLogs();
    }, 2000); // Refresh every 2 seconds
}

function loadDMPInstances() {
    fetch('/api/dmp/list')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderInstanceList(data.instances);
            }
        })
        .catch(error => console.error('Error loading DMP instances:', error));
}

function renderInstanceList(instances) {
    const listContainer = document.getElementById('instanceList');
    listContainer.innerHTML = '';

    Object.entries(instances).forEach(([dmpId, dmpData]) => {
        const item = document.createElement('div');
        item.className = `instance-item ${selectedDMP === dmpId ? 'active' : ''}`;
        item.innerHTML = `
            <div class="instance-item-title">${dmpId}</div>
            <div class="instance-item-type">${dmpData.type}</div>
        `;
        item.onclick = () => selectDMP(dmpId);
        listContainer.appendChild(item);
    });

    document.getElementById('activeDMPs').textContent = Object.keys(instances).length;
}

function selectDMP(dmpId) {
    selectedDMP = dmpId;
    loadDMPInstances();
    loadDMPParameters(dmpId);
}

function loadDMPParameters(dmpId) {
    fetch(`/api/dmp/${dmpId}/parameters`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderParameterEditor(dmpId, data.parameters);
            }
        })
        .catch(error => console.error('Error loading parameters:', error));
}

function renderParameterEditor(dmpId, parameters) {
    const editor = document.getElementById('parameterEditor');
    editor.innerHTML = `
        <div style="display: grid; gap: 0.75rem;">
            <div>
                <label>Goal:</label>
                <input type="number" id="paramGoal" value="${parameters.goal}" step="0.01" onchange="updateParamValue('goal', this.value)">
            </div>
            <div>
                <label>Initial Position (y0):</label>
                <input type="number" id="paramY0" value="${parameters.y0}" step="0.01" onchange="updateParamValue('y0', this.value)">
            </div>
            <div>
                <label>Timestep (dt):</label>
                <input type="number" id="paramDt" value="${parameters.dt}" step="0.001" onchange="updateParamValue('dt', this.value)">
            </div>
            <button class="btn btn-primary" onclick="executeDMP('${dmpId}')">Execute</button>
            <button class="btn btn-action" onclick="pauseDMP('${dmpId}')">Pause</button>
            <button class="btn btn-danger btn-small" onclick="deleteDMP('${dmpId}')">Delete</button>
        </div>
    `;
}

function updateParamValue(paramName, value) {
    if (!selectedDMP) return;
    
    const updateData = {};
    updateData[paramName] = parseFloat(value);
    
    fetch(`/api/dmp/${selectedDMP}/parameters`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updateData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLog(`Parameter updated: ${paramName} = ${value}`);
        }
    })
    .catch(error => console.error('Error updating parameter:', error));
}

function createDMPInstance() {
    document.getElementById('createModal').classList.remove('hidden');
}

function closeCreateModal() {
    document.getElementById('createModal').classList.add('hidden');
}

function handleCreateDMP(event) {
    event.preventDefault();
    
    const dmpData = {
        dmp_id: document.getElementById('dmpId').value,
        type: document.getElementById('dmpType').value,
        n_dmps: parseInt(document.getElementById('nDmps').value),
        n_bfs: parseInt(document.getElementById('nBfs').value),
        y0: parseFloat(document.getElementById('y0').value),
        goal: parseFloat(document.getElementById('goal').value)
    };

    fetch('/api/dmp/create', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dmpData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLog(`DMP created: ${data.dmp_id}`);
            closeCreateModal();
            loadDMPInstances();
            document.getElementById('createDMPForm').reset();
        }
    })
    .catch(error => console.error('Error creating DMP:', error));
}

function executeDMP(dmpId) {
    fetch(`/api/dmp/${dmpId}/execute`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addLog(`Executing DMP: ${dmpId}`);
                document.getElementById('executionState').textContent = 'executing';
            }
        })
        .catch(error => console.error('Error executing DMP:', error));
}

function pauseDMP(dmpId) {
    fetch(`/api/dmp/${dmpId}/pause`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addLog(`DMP paused: ${dmpId}`);
                document.getElementById('executionState').textContent = 'paused';
            }
        })
        .catch(error => console.error('Error pausing DMP:', error));
}

function deleteDMP(dmpId) {
    if (confirm(`Delete DMP '${dmpId}'?`)) {
        fetch(`/api/dmp/${dmpId}/delete`, { method: 'DELETE' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLog(`DMP deleted: ${dmpId}`);
                    loadDMPInstances();
                    selectedDMP = null;
                }
            })
            .catch(error => console.error('Error deleting DMP:', error));
    }
}

function executeAll() {
    addLog('Executing all DMP instances');
}

function pauseAll() {
    addLog('Pausing all DMP instances');
}

function resetAll() {
    addLog('Resetting all DMP instances');
}

function loadTrajectory() {
    fetch('/api/trajectory')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateTrajectoryChart(data.trajectory);
            }
        })
        .catch(error => console.error('Error loading trajectory:', error));
}

function updateTrajectoryChart(trajectoryData) {
    if (!trajectoryChart || !trajectoryData) return;
    
    const pointCount = Math.min(trajectoryData.length, 100);
    const labels = Array.from({length: pointCount}, (_, i) => i);
    
    trajectoryChart.data.labels = labels;
    trajectoryChart.data.datasets[0].data = trajectoryData.slice(0, pointCount);
    trajectoryChart.update();
}

function updateSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('systemStatus').textContent = 'Active';
            document.getElementById('executionState').textContent = data.execution_state;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => console.error('Error updating status:', error));
}

function loadLogs() {
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderLogs(data.logs);
            }
        })
        .catch(error => console.error('Error loading logs:', error));
}

function renderLogs(logs) {
    const logsPanel = document.getElementById('logsPanel');
    logsPanel.innerHTML = logs.reverse().slice(0, 10).map(log => {
        const timestamp = new Date(log.timestamp).toLocaleTimeString();
        const message = log.message || (typeof log.data === 'string' ? log.data : JSON.stringify(log.data));
        return `<p class="log-entry">[${timestamp}] ${message}</p>`;
    }).join('');
}

function addLog(message) {
    const logsPanel = document.getElementById('logsPanel');
    const entry = document.createElement('p');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logsPanel.insertBefore(entry, logsPanel.firstChild);
    
    // Keep only last 10 logs
    while (logsPanel.children.length > 10) {
        logsPanel.removeChild(logsPanel.lastChild);
    }
}

window.addEventListener('beforeunload', () => {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
});
