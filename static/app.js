// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resetBtn = document.getElementById('resetBtn');

// Result elements
const predictionClass = document.getElementById('predictionClass');
const confidence = document.getElementById('confidence');
const originalImage = document.getElementById('originalImage');
const heatmapImage = document.getElementById('heatmapImage');
const overlayImage = document.getElementById('overlayImage');
const probabilityBars = document.getElementById('probabilityBars');

// API endpoint
const API_URL = window.location.origin;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
resetBtn.addEventListener('click', resetApp);

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

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file upload and prediction
async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG, PNG)');
        return;
    }

    // Show loading state
    uploadArea.classList.add('hidden');
    loading.classList.add('active');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Call prediction API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error('Prediction failed');
        }

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction. Please try again.');
        resetApp();
    }
}

// Display prediction results
function displayResults(data) {
    const { prediction, images } = data;

    // Update prediction text
    predictionClass.textContent = prediction.class.toUpperCase();
    confidence.textContent = `${(prediction.confidence * 100).toFixed(2)}% Confidence`;

    // Update images
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
    // Reset file input
    fileInput.value = '';

    // Clear images
    originalImage.src = '';
    heatmapImage.src = '';
    overlayImage.src = '';

    // Clear results
    predictionClass.textContent = '-';
    confidence.textContent = '-';
    probabilityBars.innerHTML = '';

    // Reset visibility
    resultsSection.classList.remove('active');
    loading.classList.remove('active');
    uploadSection.classList.remove('hidden');
    uploadArea.classList.remove('hidden');

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
