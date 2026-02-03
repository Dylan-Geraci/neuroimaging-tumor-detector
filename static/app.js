// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resetBtn = document.getElementById('resetBtn');

// File preview elements
const filePreview = document.getElementById('filePreview');
const fileList = document.getElementById('fileList');
const clearBtn = document.getElementById('clearBtn');
const uploadBtn = document.getElementById('uploadBtn');
const addMoreBtn = document.getElementById('addMoreBtn');

// Result elements
const predictionClass = document.getElementById('predictionClass');
const confidence = document.getElementById('confidence');
const originalImage = document.getElementById('originalImage');
const heatmapImage = document.getElementById('heatmapImage');
const overlayImage = document.getElementById('overlayImage');
const probabilityBars = document.getElementById('probabilityBars');

// Batch result elements
const batchInfo = document.getElementById('batchInfo');
const scanCount = document.getElementById('scanCount');
const agreementIndicator = document.getElementById('agreementIndicator');
const individualScans = document.getElementById('individualScans');
const expandBtn = document.getElementById('expandBtn');
const scansContainer = document.getElementById('scansContainer');
const singleVisualizations = document.getElementById('singleVisualizations');
const aggregatedPrediction = document.getElementById('aggregatedPrediction');

// Upload progress elements
const uploadProgress = document.getElementById('uploadProgress');
const uploadText = document.getElementById('uploadText');
const progressBarFill = document.getElementById('progressBarFill');
const progressPercent = document.getElementById('progressPercent');
const analysisSpinner = document.getElementById('analysisSpinner');
const analysisText = document.getElementById('analysisText');

// API endpoint
const API_URL = window.location.origin;

// Toast & banner elements
const toastContainer = document.getElementById('toastContainer');
const healthBanner = document.getElementById('healthBanner');
const healthBannerMessage = document.getElementById('healthBannerMessage');

// Show an inline toast notification
function showToast(message, type = 'error', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast${type === 'warning' ? ' toast-warning' : ''}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close">&times;</button>
    `;
    toast.querySelector('.toast-close').addEventListener('click', () => removeToast(toast));
    toastContainer.appendChild(toast);
    if (duration > 0) {
        setTimeout(() => removeToast(toast), duration);
    }
}

// Animate-out then remove a toast element
function removeToast(toast) {
    if (!toast.parentNode) return;
    toast.classList.add('toast-removing');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
}

// Show the persistent health banner
function showHealthBanner(message) {
    healthBannerMessage.textContent = message;
    healthBanner.classList.remove('hidden');
}

// Dismiss the health banner
function dismissHealthBanner() {
    healthBanner.classList.add('hidden');
}

// Extract a human-readable error message from a failed API response
async function extractErrorMessage(response, fallback) {
    try {
        const body = await response.json();
        return body.detail || fallback;
    } catch {
        return fallback;
    }
}

// Store selected files
let selectedFiles = [];

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
resetBtn.addEventListener('click', resetApp);
clearBtn.addEventListener('click', clearSelection);
uploadBtn.addEventListener('click', uploadFiles);
expandBtn.addEventListener('click', toggleIndividualScans);
addMoreBtn.addEventListener('click', () => fileInput.click());

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (files.length > 0) {
        addFiles(files);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const files = Array.from(e.target.files).filter(f => {
        // Check MIME type or file extension for images
        const isImageType = f.type.startsWith('image/');
        const hasImageExtension = /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name);
        return isImageType || hasImageExtension;
    });

    if (files.length > 0) {
        addFiles(files);
    }
    // Reset input so selecting same files again triggers change event
    e.target.value = '';
}

// Add files to selection
function addFiles(files) {
    selectedFiles = [...selectedFiles, ...files];
    console.log('Total files now:', selectedFiles.length, selectedFiles.map(f => f.name));
    displayFilePreview();
    // DON'T auto-upload - user must click the button
}

// Display file preview
function displayFilePreview() {
    const fileCountEl = document.getElementById('fileCount');

    if (selectedFiles.length === 0) {
        filePreview.classList.remove('active');
        fileCountEl.textContent = '0';
        return;
    }

    fileCountEl.textContent = selectedFiles.length;
    fileList.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';

        // Create thumbnail
        const thumbnail = document.createElement('div');
        thumbnail.className = 'file-thumbnail';
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        thumbnail.appendChild(img);

        // File info
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-item-info';
        fileInfo.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
        `;

        // Remove button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'file-remove';
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = () => removeFile(index);

        fileItem.appendChild(thumbnail);
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(removeBtn);
        fileItem.style.animation = `fadeUp 0.3s cubic-bezier(0.22, 1, 0.36, 1) ${index * 0.03}s both`;
        fileList.appendChild(fileItem);
    });

    filePreview.classList.add('active');
    uploadBtn.textContent = selectedFiles.length === 1 ? 'Analyze Scan' : `Analyze ${selectedFiles.length} Scans`;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Remove file from selection
function removeFile(index) {
    selectedFiles.splice(index, 1);
    displayFilePreview();
}

// Clear selection
function clearSelection() {
    selectedFiles = [];
    fileInput.value = '';
    filePreview.classList.remove('active');
}

// --- Upload progress helpers ---

function showUploadPhase(fileCount) {
    const label = fileCount === 1 ? 'Uploading scan...' : `Uploading ${fileCount} scans...`;
    uploadText.textContent = label;
    progressBarFill.style.width = '0%';
    progressPercent.textContent = '0%';
    loading.classList.add('active', 'phase-upload');
    loading.classList.remove('phase-analyze');
}

function updateProgress(percent) {
    const clamped = Math.min(100, Math.max(0, Math.round(percent)));
    progressBarFill.style.width = clamped + '%';
    progressPercent.textContent = clamped + '%';
}

function showAnalyzePhase() {
    const fileCount = selectedFiles.length;
    const label = fileCount === 1 ? 'Analyzing scan...' : `Analyzing ${fileCount} scans...`;
    analysisText.textContent = label;
    loading.classList.remove('phase-upload');
    loading.classList.add('phase-analyze');
}

function hideLoading() {
    loading.classList.remove('active', 'phase-upload', 'phase-analyze');
}

// Promise wrapper around XMLHttpRequest with upload progress
function xhrUpload(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url);

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total) * 100;
                onProgress(percent);
            }
        });

        xhr.addEventListener('load', () => {
            let data;
            try {
                data = JSON.parse(xhr.responseText);
            } catch {
                reject(new Error('Invalid response from server'));
                return;
            }
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(data);
            } else {
                reject(new Error(data.detail || `Request failed (${xhr.status})`));
            }
        });

        xhr.addEventListener('error', () => {
            reject(new Error('Network error — please check your connection.'));
        });

        xhr.send(formData);
    });
}

// Upload files
async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    console.log('Uploading files:', selectedFiles.length, selectedFiles.map(f => f.name));

    // Hide upload UI, show progress bar
    uploadArea.classList.add('hidden');
    filePreview.classList.remove('active');
    showUploadPhase(selectedFiles.length);

    // Prepare form data & pick endpoint
    const formData = new FormData();
    let endpoint;

    if (selectedFiles.length === 1) {
        formData.append('file', selectedFiles[0]);
        endpoint = `${API_URL}/predict`;
    } else {
        selectedFiles.forEach(file => formData.append('files', file));
        endpoint = `${API_URL}/predict/batch`;
    }

    try {
        const data = await xhrUpload(endpoint, formData, updateProgress);

        // Switch to analysis phase with a brief delay so user sees it
        showAnalyzePhase();
        await new Promise(r => setTimeout(r, 400));

        if (data.success) {
            if (selectedFiles.length === 1) {
                displaySingleResult(data);
            } else {
                displayBatchResults(data);
            }
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message || 'An error occurred during prediction. Please try again.');
        resetApp();
    }
}

// Display single scan result
function displaySingleResult(data) {
    const { prediction, images } = data;

    // Update prediction text - change header for single scan
    aggregatedPrediction.querySelector('h2').textContent = 'Diagnosis';
    predictionClass.textContent = formatClassName(prediction.class);
    confidence.textContent = `${(prediction.confidence * 100).toFixed(2)}% Confidence`;

    // Hide batch-specific elements
    batchInfo.classList.add('hidden');
    individualScans.classList.add('hidden');

    // Show single visualizations
    singleVisualizations.classList.remove('hidden');
    originalImage.src = images.original;
    heatmapImage.src = images.heatmap;
    overlayImage.src = images.overlay;

    // Create probability bars
    createProbabilityBars(prediction.probabilities);

    // Hide loading, show results
    hideLoading();
    uploadSection.classList.add('hidden');
    resultsSection.classList.add('active');

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display batch results
function displayBatchResults(data) {
    console.log('Batch results received:', data);
    const { aggregated_prediction, individual_predictions, processed_count } = data;
    console.log('Individual predictions count:', individual_predictions.length);

    // Update aggregated prediction
    aggregatedPrediction.querySelector('h2').textContent = 'Aggregated Diagnosis';
    predictionClass.textContent = formatClassName(aggregated_prediction.class);
    confidence.textContent = `${(aggregated_prediction.confidence * 100).toFixed(2)}% Confidence`;

    // Show batch info
    batchInfo.classList.remove('hidden');
    scanCount.textContent = `${processed_count} scans analyzed`;

    // Agreement indicator
    const agreement = aggregated_prediction.agreement_score;
    agreementIndicator.textContent = `${(agreement * 100).toFixed(0)}% Agreement`;
    agreementIndicator.className = 'agreement-indicator';
    if (agreement === 1) {
        agreementIndicator.classList.add('full-agreement');
    } else if (agreement >= 0.75) {
        agreementIndicator.classList.add('high-agreement');
    } else if (agreement >= 0.5) {
        agreementIndicator.classList.add('medium-agreement');
    } else {
        agreementIndicator.classList.add('low-agreement');
    }

    // Create aggregated probability bars
    createProbabilityBars(aggregated_prediction.probabilities);

    // Hide single visualizations
    singleVisualizations.classList.add('hidden');

    // Show individual scans section
    individualScans.classList.remove('hidden');
    displayIndividualScans(individual_predictions);

    // Hide loading, show results
    hideLoading();
    uploadSection.classList.add('hidden');
    resultsSection.classList.add('active');

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display individual scan results
function displayIndividualScans(predictions) {
    console.log('Displaying individual scans:', predictions.length);
    scansContainer.innerHTML = '';

    predictions.forEach((pred, index) => {
        if (pred.error) {
            // Show error card
            const errorCard = document.createElement('div');
            errorCard.className = 'scan-card error';
            errorCard.innerHTML = `
                <div class="scan-header">
                    <span class="scan-filename">${pred.filename}</span>
                    <span class="scan-error">Error: ${pred.error}</span>
                </div>
            `;
            scansContainer.appendChild(errorCard);
            return;
        }

        const scanCard = document.createElement('div');
        scanCard.className = 'scan-card';
        scanCard.style.animationDelay = `${index * 0.06}s`;

        scanCard.innerHTML = `
            <div class="scan-thumbnail">
                <img src="${pred.images.original}" alt="${pred.filename}" />
            </div>
            <div class="scan-details">
                <div class="scan-filename">${pred.filename}</div>
                <div class="scan-prediction">${formatClassName(pred.class)}</div>
                <div class="scan-confidence">${(pred.confidence * 100).toFixed(1)}%</div>
            </div>
            <div class="scan-overlay">
                <img src="${pred.images.overlay}" alt="Overlay" />
            </div>
        `;

        scansContainer.appendChild(scanCard);
    });
}

// Toggle individual scans visibility
function toggleIndividualScans() {
    scansContainer.classList.toggle('expanded');
    expandBtn.classList.toggle('expanded');

    const btnText = expandBtn.querySelector('span');
    if (scansContainer.classList.contains('expanded')) {
        btnText.textContent = 'Hide Individual Scans';
    } else {
        btnText.textContent = 'View Individual Scans';
    }
}

// Make toggle function globally accessible for fallback
window.toggleIndividualScans = toggleIndividualScans;

// Create probability bar visualizations
function createProbabilityBars(probabilities) {
    probabilityBars.innerHTML = '';

    // Sort by probability (descending)
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([className, prob], index) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';

        const label = document.createElement('div');
        label.className = 'prob-label';
        const labelText = document.createElement('span');
        labelText.textContent = formatClassName(className);
        label.appendChild(labelText);

        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar-container';

        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.width = '0%'; // Start at 0 for animation

        const value = document.createElement('div');
        value.className = 'prob-value';
        value.textContent = `${(prob * 100).toFixed(2)}%`;

        barContainer.appendChild(bar);
        label.appendChild(value);
        probItem.appendChild(label);
        probItem.appendChild(barContainer);
        probabilityBars.appendChild(probItem);

        // Animate bar with stagger
        setTimeout(() => {
            bar.style.width = `${prob * 100}%`;
        }, 100 + (index * 80));
    });
}

// Format class name for display
function formatClassName(name) {
    const nameMap = {
        'glioma': 'Glioma',
        'meningioma': 'Meningioma',
        'notumor': 'No Tumor',
        'pituitary': 'Pituitary Tumor'
    };
    return nameMap[name] || name;
}

// Reset application
function resetApp() {
    // Reset file input and selection
    fileInput.value = '';
    selectedFiles = [];

    // Clear images
    originalImage.src = '';
    heatmapImage.src = '';
    overlayImage.src = '';

    // Clear results
    predictionClass.textContent = '-';
    confidence.textContent = '-';
    probabilityBars.innerHTML = '';
    scansContainer.innerHTML = '';

    // Reset visibility
    resultsSection.classList.remove('active');
    hideLoading();
    uploadSection.classList.remove('hidden');
    uploadArea.classList.remove('hidden');
    filePreview.classList.remove('active');
    singleVisualizations.classList.remove('hidden');
    individualScans.classList.remove('hidden');
    batchInfo.classList.remove('hidden');
    scansContainer.classList.remove('expanded');
    expandBtn.classList.remove('expanded');

    // Reset expand button text
    const btnText = expandBtn.querySelector('span');
    btnText.textContent = 'View Individual Scans';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
        showHealthBanner('Could not connect to the prediction API. Please ensure the backend is running.');
    }
}

// Initialize
checkHealth();
