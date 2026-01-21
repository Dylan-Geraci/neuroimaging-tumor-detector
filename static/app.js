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

// API endpoint
const API_URL = window.location.origin;

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
    const rawCount = e.target.files.length;
    const debugInfo = document.getElementById('debugInfo');
    debugInfo.textContent = `Files from dialog: ${rawCount}`;

    const files = Array.from(e.target.files).filter(f => {
        // Check MIME type or file extension for images
        const isImageType = f.type.startsWith('image/');
        const hasImageExtension = /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name);
        return isImageType || hasImageExtension;
    });

    debugInfo.textContent = `Files from dialog: ${rawCount}, After filter: ${files.length}`;

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

// Upload files
async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    console.log('Uploading files:', selectedFiles.length, selectedFiles.map(f => f.name));

    // Show loading state
    uploadArea.classList.add('hidden');
    filePreview.classList.remove('active');
    loading.classList.add('active');

    // Prepare form data
    const formData = new FormData();

    if (selectedFiles.length === 1) {
        // Single file - use original endpoint
        formData.append('file', selectedFiles[0]);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                displaySingleResult(data);
            } else {
                throw new Error('Prediction failed');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please try again.');
            resetApp();
        }
    } else {
        // Multiple files - use batch endpoint
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch(`${API_URL}/predict/batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                displayBatchResults(data);
            } else {
                throw new Error('Batch prediction failed');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during batch prediction. Please try again.');
            resetApp();
        }
    }
}

// Display single scan result
function displaySingleResult(data) {
    const { prediction, images } = data;

    // Update prediction text - change header for single scan
    aggregatedPrediction.querySelector('h2').textContent = 'Diagnosis';
    predictionClass.textContent = prediction.class.toUpperCase();
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
    loading.classList.remove('active');
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
    predictionClass.textContent = aggregated_prediction.class.toUpperCase();
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
    loading.classList.remove('active');
    uploadSection.classList.add('hidden');
    resultsSection.classList.add('active');

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display individual scan results
function displayIndividualScans(predictions) {
    console.log('Displaying individual scans:', predictions.length);
    scansContainer.innerHTML = '';

    predictions.forEach(pred => {
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
    console.log('Toggle clicked, current expanded state:', scansContainer.classList.contains('expanded'));
    scansContainer.classList.toggle('expanded');
    expandBtn.classList.toggle('expanded');

    const btnText = expandBtn.querySelector('span');
    if (scansContainer.classList.contains('expanded')) {
        btnText.textContent = 'Hide Individual Scans';
    } else {
        btnText.textContent = 'View Individual Scans';
    }
    console.log('New expanded state:', scansContainer.classList.contains('expanded'));
}

// Make toggle function globally accessible for fallback
window.toggleIndividualScans = toggleIndividualScans;

// Create probability bar visualizations
function createProbabilityBars(probabilities) {
    probabilityBars.innerHTML = '';

    // Sort by probability (descending)
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([className, prob]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';

        const label = document.createElement('div');
        label.className = 'prob-label';
        label.textContent = formatClassName(className);

        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar-container';

        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.width = '0%'; // Start at 0 for animation

        const value = document.createElement('div');
        value.className = 'prob-value';
        value.textContent = `${(prob * 100).toFixed(2)}%`;

        barContainer.appendChild(bar);
        probItem.appendChild(label);
        probItem.appendChild(barContainer);
        probItem.appendChild(value);
        probabilityBars.appendChild(probItem);

        // Animate bar
        setTimeout(() => {
            bar.style.width = `${prob * 100}%`;
        }, 100);
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
    loading.classList.remove('active');
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
        alert('Warning: Could not connect to the prediction API. Please ensure the backend is running.');
    }
}

// Initialize
checkHealth();
