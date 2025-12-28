// Wait for the document to be fully loaded
document.addEventListener("DOMContentLoaded", () => {

    // --- Theme Toggle Logic ---
    const themeToggleButton = document.getElementById("theme-toggle");
    const sunIcon = document.getElementById("sun-icon");
    const moonIcon = document.getElementById("moon-icon");

    // Check for saved theme in localStorage
    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark-mode");
        sunIcon.classList.add("hidden");
        moonIcon.classList.remove("hidden");
    }

    themeToggleButton.addEventListener("click", () => {
        // Toggle the .dark-mode class on the body
        document.body.classList.toggle("dark-mode");

        // Check if dark mode is now on
        if (document.body.classList.contains("dark-mode")) {
            // Save preference to localStorage
            localStorage.setItem("theme", "dark");
            // Show moon, hide sun
            sunIcon.classList.add("hidden");
            moonIcon.classList.remove("hidden");
        } else {
            // Save preference to localStorage
            localStorage.setItem("theme", "light");
            // Show sun, hide moon
            sunIcon.classList.remove("hidden");
            moonIcon.classList.add("hidden");
        }
    });


    // --- API & Form Logic ---
    const analyzeButton = document.getElementById("analyze-button");
    const fileInput = document.getElementById("file-input"); // Changed from emailInput
    const fileNameDisplay = document.getElementById("file-name"); // To show the selected file name
    const loadingSpinner = document.getElementById("loading-spinner");
    const resultBox = document.getElementById("result-box");
    const resultTitle = document.getElementById("result-title");
    const resultConfidence = document.getElementById("result-confidence");

    // CRITICAL CHECK: Ensure all required HTML elements are found.
    // Added fileInput and fileNameDisplay checks
    if (!analyzeButton || !fileInput || !fileNameDisplay || !loadingSpinner || !resultBox || !resultTitle || !resultConfidence) {
        console.error("CRITICAL ERROR: One or more required HTML elements (e.g., analyze-button, file-input, result-box) were not found. Check templates/index.html for correct IDs.");
        return; // Stop if elements are missing
    }

    // Function to set the result box to an error state
    function showError(message) {
        resultBox.style.display = "block";
        resultBox.classList.remove("result-legitimate");
        resultBox.classList.add("result-phishing"); // Use phishing style for general errors
        resultTitle.innerText = "Error"; // Changed from "Attention"
        resultConfidence.innerText = message;
    }

    // Clear the result box display
    function clearResults() {
        resultBox.style.display = "none";
        resultBox.classList.remove("result-phishing", "result-legitimate");
        resultTitle.innerText = "";
        resultConfidence.innerText = "";
    }

    // --- File Input Logic ---
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            // Basic check for .txt extension
            if (file.name.toLowerCase().endsWith('.txt')) {
                fileNameDisplay.textContent = file.name;
                clearResults(); // Clear results when a new file is chosen
            } else {
                fileNameDisplay.textContent = 'Please select a .txt file';
                fileInput.value = ''; // Reset the input if not a .txt file
                showError('Invalid file type. Only .txt files are accepted.');
            }
        } else {
            fileNameDisplay.textContent = 'No file selected...';
            clearResults();
        }
    });

    // --- Analyze Button Click Logic ---
    analyzeButton.addEventListener("click", () => {
        const file = fileInput.files[0]; // Get the selected file
        clearResults(); // Clear any previous results/errors

        // 1. Check if a file is selected
        if (!file) {
            showError("Please choose a .txt file to analyze.");
            return;
        }

        // 2. Check if it's a .txt file (redundant check, but safe)
        if (!file.name.toLowerCase().endsWith('.txt') || file.type !== "text/plain") {
             // Check MIME type as well for robustness
            showError("Invalid file type selected. Please upload only .txt files.");
            return;
        }


        // Show loading, hide previous results
        loadingSpinner.style.display = "block";
        analyzeButton.disabled = true;

        // 3. Read the file content
        const reader = new FileReader();

        reader.onload = (e) => {
            // This runs when the file is successfully read
            const text = e.target.result; // Get the text content from the file

            // Log the start of the request
            console.log("File read successfully. Sending text to /predict (length:", text.length + ")");

            // 4. Send the TEXT content to the Flask backend
            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: text }), // Send the file content as 'text'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                console.log("Received successful response from server.");
                return response.json();
            })
            .then(data => {
                console.log("Prediction data received:", data);

                loadingSpinner.style.display = "none";
                analyzeButton.disabled = false;

                if (data.prediction === undefined || data.confidence === undefined) {
                    console.error("API response is missing required fields.", data);
                    showError("Invalid data format received from AI server.");
                    return;
                }

                // Display results
                resultBox.style.display = "block";
                resultBox.classList.remove("result-phishing", "result-legitimate");

                if (data.prediction === "Phishing") {
                    resultTitle.innerText = "Phishing Risk";
                    resultBox.classList.add("result-phishing");
                } else {
                    resultTitle.innerText = "Legitimate";
                    resultBox.classList.add("result-legitimate");
                }

                resultConfidence.innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            })
            .catch(error => {
                loadingSpinner.style.display = "none";
                analyzeButton.disabled = false;

                let errorMessage = "Could not connect to the analysis server. Please check the backend log.";
                if (error.message.includes("HTTP error")) {
                    errorMessage = `Server Error: ${error.message}`;
                }

                console.error("Fetch Error (Check Backend Status):", error);
                showError(errorMessage);
            });
        };

        reader.onerror = () => {
            // This runs if there's an error reading the file
            loadingSpinner.style.display = "none";
            analyzeButton.disabled = false;
            showError("Could not read the selected .txt file.");
            console.error("FileReader error");
        };

        // Start reading the file as text
        reader.readAsText(file);
    });
});

