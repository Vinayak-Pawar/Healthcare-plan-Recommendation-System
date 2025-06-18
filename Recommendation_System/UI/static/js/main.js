// Healthcare Recommendation System - JavaScript

class HealthcareUI {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupTabs();
        console.log('🏥 Healthcare Recommendation System UI initialized');
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        const searchBtn = document.getElementById('searchBtn');
        
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce(this.searchPatients.bind(this), 300));
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.searchPatients();
                }
            });
        }

        if (searchBtn) {
            searchBtn.addEventListener('click', this.searchPatients.bind(this));
        }

        // Generate plan button
        const generateBtn = document.getElementById('generatePlan');
        if (generateBtn) {
            generateBtn.addEventListener('click', this.generateHealthPlan.bind(this));
        }
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach((button, index) => {
            button.addEventListener('click', () => {
                // Remove active class from all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab
                button.classList.add('active');
                if (tabContents[index]) {
                    tabContents[index].classList.add('active');
                }
            });
        });
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    showLoading(element) {
        element.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
        `;
    }

    showHealthPlanLoading(element) {
        element.innerHTML = `
            <div class="health-plan-loading" style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; margin: 2rem 0;">
                <div class="loading-spinner-container" style="margin-bottom: 2rem;">
                    <div class="medical-spinner" style="width: 80px; height: 80px; margin: 0 auto; position: relative;">
                        <div class="spinner-ring" style="width: 80px; height: 80px; border: 4px solid rgba(255,255,255,0.3); border-top: 4px solid #ffffff; border-radius: 50%; animation: spin 1s linear infinite; position: absolute;"></div>
                        <div class="spinner-pulse" style="width: 60px; height: 60px; background: rgba(255,255,255,0.2); border-radius: 50%; position: absolute; top: 10px; left: 10px; animation: pulse 2s ease-in-out infinite;"></div>
                        <div class="medical-cross" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 24px;">🏥</div>
                    </div>
                </div>
                
                <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600;">
                    🤖 Generating AI Health Plan
                </h3>
                
                <div class="loading-stages" style="max-width: 500px; margin: 0 auto;">
                    <div class="stage active" id="stage-1" style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 4px solid #4CAF50;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <span class="stage-icon">📋</span>
                            <span>Analyzing patient medical history...</span>
                            <div class="stage-spinner" style="width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 0.8s linear infinite; margin-left: auto;"></div>
                        </div>
                    </div>
                    
                    <div class="stage" id="stage-2" style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.3);">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <span class="stage-icon">👥</span>
                            <span>Finding similar patients...</span>
                        </div>
                    </div>
                    
                    <div class="stage" id="stage-3" style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.3);">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <span class="stage-icon">🧠</span>
                            <span>AI models analyzing data...</span>
                        </div>
                    </div>
                    
                    <div class="stage" id="stage-4" style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.3);">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <span class="stage-icon">📝</span>
                            <span>Generating personalized recommendations...</span>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; opacity: 0.9; font-size: 0.9rem;">
                    <p style="margin: 0;">🔬 Using advanced AI models: GPT-4, Claude, Gemini & Perplexity</p>
                    <p style="margin: 0.5rem 0 0 0;">💊 Checking drug safety with FDA databases</p>
                </div>
            </div>
            
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                @keyframes pulse {
                    0%, 100% { transform: scale(1); opacity: 0.7; }
                    50% { transform: scale(1.1); opacity: 0.3; }
                }
                
                .stage.active {
                    background: rgba(255,255,255,0.15) !important;
                    border-left-color: #4CAF50 !important;
                }
                
                .stage.completed {
                    background: rgba(76, 175, 80, 0.2) !important;
                    border-left-color: #4CAF50 !important;
                }
                
                .stage.completed .stage-spinner {
                    display: none;
                }
                
                .stage.completed .stage-icon:after {
                    content: " ✅";
                }
            </style>
        `;

        // Simulate stage progression
        this.simulateLoadingStages();
    }

    simulateLoadingStages() {
        const stages = [
            { id: 'stage-1', delay: 1000 },
            { id: 'stage-2', delay: 2500 },
            { id: 'stage-3', delay: 4000 },
            { id: 'stage-4', delay: 5500 }
        ];

        stages.forEach((stage, index) => {
            setTimeout(() => {
                const currentStage = document.getElementById(stage.id);
                if (currentStage) {
                    // Mark current stage as completed
                    currentStage.classList.remove('active');
                    currentStage.classList.add('completed');
                    
                    // Activate next stage
                    const nextStageId = `stage-${index + 2}`;
                    const nextStage = document.getElementById(nextStageId);
                    if (nextStage) {
                        nextStage.classList.add('active');
                        nextStage.style.background = 'rgba(255,255,255,0.1)';
                        nextStage.style.borderLeftColor = '#4CAF50';
                        
                        // Add spinner to active stage
                        const spinnerHtml = '<div class="stage-spinner" style="width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 0.8s linear infinite; margin-left: auto;"></div>';
                        const stageContent = nextStage.querySelector('div');
                        if (stageContent && !stageContent.querySelector('.stage-spinner')) {
                            stageContent.insertAdjacentHTML('beforeend', spinnerHtml);
                        }
                    }
                }
            }, stage.delay);
        });
    }

    showError(element, message) {
        element.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${message}
            </div>
        `;
    }

    showSuccess(element, message) {
        element.innerHTML = `
            <div class="alert alert-success">
                <strong>Success:</strong> ${message}
            </div>
        `;
    }

    async searchPatients() {
        const searchInput = document.getElementById('searchInput');
        const resultsContainer = document.getElementById('searchResults');
        const query = searchInput.value.trim();

        if (!query || query.length < 2) {
            resultsContainer.innerHTML = '';
            return;
        }

        this.showLoading(resultsContainer);

        try {
            const response = await fetch(`/search_patients?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (data.error) {
                this.showError(resultsContainer, data.error);
                return;
            }

            this.displaySearchResults(data.patients, resultsContainer);

        } catch (error) {
            console.error('Search error:', error);
            this.showError(resultsContainer, 'Failed to search patients');
        }
    }

    displaySearchResults(patients, container) {
        if (patients.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    No patients found matching your search criteria.
                </div>
            `;
            return;
        }

        const patientCards = patients.map(patient => `
            <div class="patient-card ${patient.is_deceased ? 'deceased' : ''}" 
                 onclick="app.selectPatient('${patient.id}', '${patient.name}')">
                <div class="patient-info">
                    <div>
                        <div class="patient-name">${patient.name}</div>
                        <div class="patient-details">
                            ID: ${patient.id} | 
                            Born: ${patient.birthdate} | 
                            Gender: ${patient.gender} |
                            Location: ${patient.city}, ${patient.state}
                        </div>
                    </div>
                    <div class="patient-status ${patient.is_deceased ? 'status-deceased' : 'status-living'}">
                        ${patient.is_deceased ? '💀 Deceased' : '✅ Living'}
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = `
            <div class="search-results">
                <h3>Search Results (${patients.length} found)</h3>
                <div class="patient-list">
                    ${patientCards}
                </div>
            </div>
        `;
    }

    selectPatient(patientId, patientName) {
        const searchInput = document.getElementById('searchInput');
        const generateBtn = document.getElementById('generatePlan');
        
        searchInput.value = patientId;
        generateBtn.disabled = false;
        generateBtn.textContent = `Generate Health Plan for ${patientName}`;
        
        // Clear search results
        document.getElementById('searchResults').innerHTML = '';
        
        // Store selected patient info
        this.selectedPatient = { id: patientId, name: patientName };
        
        console.log(`Selected patient: ${patientName} (${patientId})`);
    }

    async generateHealthPlan() {
        const searchInput = document.getElementById('searchInput');
        const generateBtn = document.getElementById('generatePlan');
        const resultsSection = document.getElementById('resultsSection');
        const resultsHeader = document.getElementById('resultsHeader');
        const resultsContent = document.getElementById('resultsContent');
        const patientId = searchInput.value.trim();

        if (!patientId) {
            alert('Please enter a patient ID or search and select a patient');
            return;
        }

        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<div class="spinner" style="width: 1rem; height: 1rem; margin-right: 0.5rem;"></div> Generating Health Plan...';
        
        // Show results section and loading
        resultsSection.style.display = 'block';
        resultsHeader.textContent = `Executing Health Plan Generation for Patient ID: ${patientId}`;
        this.showHealthPlanLoading(resultsContent);

        try {
            // Call the new endpoint to run r_engine.py
            const response = await fetch('/run_engine', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ patient_id: patientId })
            });

            const data = await response.json();

            if (data.error) {
                this.showError(resultsContent, data.error);
                return;
            }

            // Display successful execution results
            this.displayEngineResults(data, resultsContent, resultsHeader);

        } catch (error) {
            console.error('Engine execution error:', error);
            this.showError(resultsContent, 'Failed to execute r_engine.py program');
        } finally {
            // Reset button
            generateBtn.disabled = false;
            generateBtn.textContent = '🤖 Generate Health Plan';
        }
    }

    displayEngineResults(data, container, header) {
        header.textContent = `✅ Health Plan Generation Completed for ${data.patient.name}`;
        
        const resultsHtml = `
            <div class="results-tabs">
                <button class="tab-button active" data-tab="current">📋 Current Status</button>
                <button class="tab-button" data-tab="recommendations">🤖 AI Recommendations</button>
                <button class="tab-button" data-tab="similar">👥 Similar Patients</button>
                <button class="tab-button" data-tab="files">📁 Generated Files</button>
            </div>

            <div class="tab-content active" id="current-tab">
                <h3>📋 Current Patient Status</h3>
                <div class="content-box" style="background: #ffffff; max-height: 400px; overflow-y: auto;">
                    ${data.current_status || 'No current status available.'}
                </div>
                ${data.files.current_status ? `
                    <div class="file-downloads">
                        <a href="/files/Output/${data.files.current_status}" class="download-btn">
                            📄 Download Current Status Report
                        </a>
                    </div>
                ` : ''}
            </div>

            <div class="tab-content" id="recommendations-tab">
                <h3>🤖 AI-Generated Health Plan</h3>
                <div class="content-box" style="background: #ffffff; max-height: 600px; overflow-y: auto; white-space: pre-wrap; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 14px; line-height: 1.6;">
                    ${this.displayFuturePlan(data)}
                </div>
                ${data.files.future_plan ? `
                    <div class="file-downloads">
                        <a href="/files/Future_Suggestions/${data.files.future_plan}" class="download-btn">
                            📄 Download Complete Health Plan
                        </a>
                    </div>
                ` : ''}
            </div>

            <div class="tab-content" id="similar-tab">
                <h3>👥 Similar Patients Analysis</h3>
                ${this.displaySimilarPatients(data)}
            </div>

            <div class="tab-content" id="files-tab">
                <h3>📁 Generated Files</h3>
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1.5rem;">
                    <p style="color: #6c757d; margin-bottom: 1rem;">
                        The recommendation system has generated the following files:
                    </p>
                    <div style="display: grid; gap: 1rem;">
                        ${data.files.current_status ? `
                            <div style="padding: 1rem; border: 1px solid #dee2e6; border-radius: 4px;">
                                <h4>📋 Current Status Report</h4>
                                <p>File: ${data.files.current_status}</p>
                                <a href="/files/Output/${data.files.current_status}" class="btn btn-info btn-sm">Download</a>
                            </div>
                        ` : ''}
                        ${data.files.future_plan ? `
                            <div style="padding: 1rem; border: 1px solid #dee2e6; border-radius: 4px;">
                                <h4>🤖 AI Recommendations</h4>
                                <p>File: ${data.files.future_plan}</p>
                                <a href="/files/Future_Suggestions/${data.files.future_plan}" class="btn btn-info btn-sm">Download</a>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = resultsHtml;
        
        // Setup new tabs
        this.setupTabs();
    }

    displayFuturePlan(data) {
        // If we have the full future plan content, display it
        if (data.future_plan && data.future_plan.length > 100) {
            return data.future_plan;
        }
        
        // Otherwise, show a message to download the file
        return "Future Health Plan has been generated successfully.\n\nPlease download the complete file using the download button below to view the full AI-generated health plan.";
    }

    displaySimilarPatients(data) {
        if (!data.similar_patient_files || data.similar_patient_files.length === 0) {
            return `
                <div class="alert alert-info">
                    <p>No similar patients were analyzed for this case.</p>
                </div>
            `;
        }

        const similarPatientsHtml = data.similar_patient_files.map((patient, index) => `
            <div style="padding: 1rem; border: 1px solid #dee2e6; border-radius: 8px; margin-bottom: 1rem; background: #ffffff;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: #2c3e50;">
                        👤 ${patient.patient_name}
                    </h4>
                    <span style="background: #e8f5e8; color: #2d5a2d; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">
                        ${Math.round(patient.similarity_score * 100)}% Match
                    </span>
                </div>
                <div style="color: #6c757d; font-size: 0.9rem; margin-bottom: 1rem;">
                    <strong>Patient ID:</strong> ${patient.patient_id}
                </div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <a href="/files/Output/${patient.filename}" class="btn btn-info btn-sm">
                        📄 Download Status Report
                    </a>
                    <span style="color: #6c757d; font-size: 0.85rem;">
                        File: ${patient.filename}
                    </span>
                </div>
            </div>
        `).join('');

        return `
            <div style="margin-bottom: 1rem;">
                <p style="color: #6c757d;">
                    Found <strong>${data.similar_patient_files.length}</strong> similar patients based on medical history, conditions, and demographics.
                    These patients' status reports were analyzed to generate personalized recommendations.
                </p>
            </div>
            <div>
                ${similarPatientsHtml}
            </div>
        `;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new HealthcareUI();
});

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    }).catch(() => {
        alert('Failed to copy to clipboard');
    });
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
} 