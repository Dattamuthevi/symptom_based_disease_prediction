document.addEventListener("DOMContentLoaded", function() {
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;
    const chatBox = document.getElementById("chat-box");
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-input");
    const chatHistory = document.getElementById("chat-history");

    // **1️⃣ Toggle Dark/Light Mode**
    themeToggle.addEventListener("click", function() {
        body.classList.toggle("dark-mode");
        themeToggle.innerHTML = body.classList.contains("dark-mode") ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    });

    // **2️⃣ Send Message on Click or Enter**
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function(e) {
        if (e.key === "Enter") sendMessage();
    });

    function sendMessage() {
        let symptom = userInput.value.trim().toLowerCase();
        if (symptom === "") return;

        // **3️⃣ Display User's Message in Chat**
        appendMessage(symptom, "user");

        // **4️⃣ Predict Disease & Response (Simulated)**
        let response = getPredictedResponse(symptom);
        appendMessage(response, "bot");

        // **5️⃣ Save to Chat History**
        let historyItem = document.createElement("li");
        historyItem.textContent = symptom;
        chatHistory.appendChild(historyItem);

        userInput.value = ""; // Clear input
    }

    function appendMessage(text, sender) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = `<p>${text}</p>`;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to latest message
    }

    // **6️⃣ Simulated Disease Prediction**
    function getPredictedResponse(symptom) {
        const diseases = {
            "sneezing": {
                disease: "COMMON COLD",
                confidence: "83%",
                medication: ["Cetirizine (10mg once daily)", "Loratadine, Fexofenadine"],
                diet: ["Increase Vitamin C: Citrus fruits (Oranges, Lemons)", "Warm ginger honey tea", "Stay hydrated with warm fluids"],
                advice: "If symptoms persist beyond 7 days, consult a physician."
            },
            "fever": {
                disease: "VIRAL FEVER",
                confidence: "90%",
                medication: ["Paracetamol (500mg every 6 hours)", "Ibuprofen (only if no stomach issues)"],
                diet: ["Drink electrolyte-rich fluids", "Consume light soups and porridge"],
                advice: "Monitor temperature. If fever exceeds 102°F for more than 48 hours, seek medical attention."
            },x
            "cough": {
                disease: "BRONCHITIS",
                confidence: "78%",
                medication: ["Dextromethorphan (Cough Suppressant)", "Guaifenesin (Expectorant)"],
                diet: ["Honey with warm water", "Avoid cold and dairy products"],
                advice: "Use a humidifier. If symptoms persist beyond 2 weeks, visit a doctor."
            },
            "headache": {
                disease: "MIGRAINE",
                confidence: "85%",
                medication: ["Ibuprofen (200mg)", "Sumatriptan (if severe)"],
                diet: ["Reduce caffeine intake", "Avoid strong lights and noises"],
                advice: "Rest in a dark room. If headaches become frequent, consult a neurologist."
            },
            "stomach pain": {
                disease: "GASTRITIS",
                confidence: "80%",
                medication: ["Omeprazole (20mg once daily)", "Antacid syrup"],
                diet: ["Avoid spicy foods", "Consume bananas, plain yogurt"],
                advice: "Avoid alcohol & smoking. If pain worsens, get a check-up."
            },
            "fatigue": {
                disease: "IRON DEFICIENCY ANEMIA",
                confidence: "75%",
                medication: ["Ferrous Sulfate (Iron supplement)", "Vitamin B12"],
                diet: ["Green leafy vegetables", "Lean meats, eggs"],
                advice: "Get blood tests done. Increase iron-rich foods."
            },
            "vomiting": {
                disease: "FOOD POISONING",
                confidence: "88%",
                medication: ["ORS (Oral Rehydration Solution)", "Domperidone (Anti-nausea)"],
                diet: ["Drink coconut water", "Consume light meals like khichdi"],
                advice: "Stay hydrated. Seek medical help if vomiting persists beyond 24 hours."
            },
            "throat pain": {
                disease: "STREP THROAT",
                confidence: "82%",
                medication: ["Amoxicillin (Antibiotic)", "Chlorhexidine Gargle"],
                diet: ["Drink warm turmeric milk", "Avoid cold drinks"],
                advice: "If fever is present, visit a doctor for antibiotics."
            },
            "joint pain": {
                disease: "RHEUMATOID ARTHRITIS",
                confidence: "79%",
                medication: ["NSAIDs (Ibuprofen)", "Methotrexate (for severe cases)"],
                diet: ["Turmeric milk", "Omega-3 rich foods (Salmon, Flaxseeds)"],
                advice: "Do light stretching. If pain worsens, consult a rheumatologist."
            },
            "burning urination": {
                disease: "URINARY TRACT INFECTION (UTI)",
                confidence: "86%",
                medication: ["Ciprofloxacin (Antibiotic)", "Cranberry supplements"],
                diet: ["Drink plenty of water", "Avoid caffeine"],
                advice: "Complete the antibiotic course. Seek medical attention if symptoms persist."
            }
        };

        if (diseases[symptom]) {
            let data = diseases[symptom];
            return `-----------------------------------------<br>
            ✅ <b>Predicted Disease:</b> ${data.disease}<br>
            📊 <b>Confidence Score:</b> ${data.confidence}<br><br>
            💊 <b>Recommended Medication:</b><br>
            - ${data.medication.join("<br>- ")}<br><br>
            🥗 <b>Recommended Diet:</b><br>
            - ${data.diet.join("<br>- ")}<br><br>
            📌 <b>Advice:</b><br>${data.advice}<br>
            -----------------------------------------`;
        } else {
            return "❌ Sorry, I couldn't identify the disease. Try providing more symptoms.";
        }
    }
});