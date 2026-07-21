import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the Linear Regression pickle model safely
MODEL_PATH = "model_linear_regression.pkl"
SCALER_PATH = "scaler.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ValuatX AI | Real Estate Valuation Engine</title>
    <!-- Google Fonts & FontAwesome -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-void: #030712;
            --card-glass: rgba(15, 23, 42, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --cyan-glow: #00f2fe;
            --violet-glow: #7928ca;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(3, 7, 18, 0.65);
            --input-border: rgba(255, 255, 255, 0.12);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-void);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        .ambient-orb {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(140px);
            z-index: -1;
            opacity: 0.25;
            pointer-events: none;
        }
        .orb-cyan { top: -200px; right: -100px; background: var(--cyan-glow); }
        .orb-violet { bottom: -200px; left: -100px; background: var(--violet-glow); }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 22px 6%;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(3, 7, 18, 0.85);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .brand-logo {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--cyan-glow), var(--violet-glow));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.25rem;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.35);
        }

        .nav-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--cyan-glow);
            background: rgba(0, 242, 254, 0.08);
            border: 1px solid rgba(0, 242, 254, 0.25);
            padding: 6px 16px;
            border-radius: 30px;
        }

        .workspace-container {
            max-width: 1400px;
            margin: 35px auto;
            padding: 0 4%;
            width: 100%;
            flex-grow: 1;
        }

        .header-section {
            text-align: center;
            margin-bottom: 40px;
        }

        .title-badge {
            display: inline-block;
            padding: 6px 18px;
            background: rgba(121, 40, 202, 0.12);
            border: 1px solid rgba(121, 40, 202, 0.35);
            border-radius: 30px;
            color: #d8b4fe;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -1.5px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #ffffff 0%, #a5f3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .main-subtitle {
            color: var(--text-sub);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
        }

        /* Workspace Grid */
        .grid-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 32px;
            align-items: start;
        }

        /* STICKY SIDEBAR PROPERTY TO FIX SCROLLING ISSUE */
        .sticky-sidebar {
            position: sticky;
            top: 110px; /* Sticks right below the navbar */
            z-index: 10;
        }

        .glass-card {
            background: var(--card-glass);
            border-radius: 28px;
            border: 1px solid var(--card-border);
            padding: 36px;
            backdrop-filter: blur(24px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--card-border);
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .section-title i {
            color: var(--cyan-glow);
        }

        .sub-group-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--cyan-glow);
            font-weight: 800;
            margin: 24px 0 16px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .live-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan-glow);
            font-weight: 700;
        }

        input, select {
            width: 100%;
            padding: 13px 15px;
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.92rem;
            font-weight: 500;
            transition: all 0.25s ease;
        }

        select {
            appearance: none;
            cursor: pointer;
            padding-right: 36px;
        }

        .select-wrapper {
            position: relative;
        }

        .select-wrapper::after {
            content: '\f107';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            right: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-sub);
            pointer-events: none;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--cyan-glow);
            box-shadow: 0 0 0 4px rgba(0, 242, 254, 0.15);
            background-color: rgba(3, 7, 18, 0.9);
        }

        .btn-predict {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, var(--cyan-glow) 0%, var(--violet-glow) 100%);
            color: white;
            font-weight: 800;
            font-size: 1.05rem;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            box-shadow: 0 10px 30px rgba(0, 242, 254, 0.25);
            letter-spacing: 0.5px;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(0, 242, 254, 0.4);
            filter: brightness(1.1);
        }

        .valuation-card {
            background: linear-gradient(180deg, rgba(121, 40, 202, 0.15) 0%, rgba(0, 242, 254, 0.05) 100%);
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .val-tag {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-sub);
            font-weight: 700;
            margin-bottom: 8px;
        }

        .val-price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 800;
            color: #ffffff;
            margin: 10px 0;
            background: linear-gradient(to right, #00f2fe, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .feature-badge-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 24px;
        }

        .f-badge {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 12px;
            text-align: left;
        }

        .f-badge-title {
            font-size: 0.72rem;
            color: var(--text-sub);
            text-transform: uppercase;
        }

        .f-badge-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--cyan-glow);
            margin-top: 4px;
        }

        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 28px;
            color: var(--text-sub);
            font-size: 0.85rem;
            border-top: 1px solid var(--card-border);
            margin-top: auto;
        }

        @media (max-width: 1024px) {
            .grid-layout { grid-template-columns: 1fr; }
            .form-grid { grid-template-columns: repeat(2, 1fr); }
            .sticky-sidebar { position: static; } /* Reverts to normal flow on mobile */
        }

        @media (max-width: 640px) {
            .form-grid { grid-template-columns: 1fr; }
            .main-title { font-size: 2.2rem; }
        }
    </style>
</head>
<body>

<div class="ambient-orb orb-cyan"></div>
<div class="ambient-orb orb-violet"></div>

<nav class="navbar">
    <div class="brand">
        <div class="brand-logo">
            <i class="fa-solid fa-house-signal"></i>
        </div>
        <span>ValuatX<span style="color: var(--cyan-glow);">.ai</span></span>
    </div>
    <div class="nav-tag">
        <i class="fa-solid fa-microchip"></i> Linear Regression Core v1.6
    </div>
</nav>

<div class="workspace-container">
    <div class="header-section">
        <span class="title-badge">Real Estate Valuation</span>
        <h1 class="main-title">Automated Valuation Model</h1>
        <p class="main-subtitle">Synthesize 16 architectural, regional, and temporal parameters into an instant real-time market value estimate.</p>
    </div>

    <div class="grid-layout">
        
        <!-- Left: Form Inputs -->
        <div class="glass-card">
            <div class="section-header">
                <span class="section-title"><i class="fa-solid fa-sliders"></i> Property Attributes</span>
                <span style="font-size: 0.8rem; color: var(--text-sub);">16 Metrics Configured</span>
            </div>

            <form id="valuationForm">
                
                <div class="sub-group-title"><i class="fa-solid fa-ruler-combined"></i> Dimensions & Layout</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Bedrooms <span class="live-val" id="bedVal">3</span></label>
                        <input type="number" name="number of bedrooms" value="3" min="1" max="20" required oninput="document.getElementById('bedVal').textContent = this.value">
                    </div>

                    <div class="form-group">
                        <label>Bathrooms <span class="live-val" id="bathVal">2.25</span></label>
                        <input type="number" step="0.25" name="number of bathrooms" value="2.25" min="1" max="10" required oninput="document.getElementById('bathVal').textContent = this.value">
                    </div>

                    <div class="form-group">
                        <label>Floors <span class="live-val" id="floorVal">1.5</span></label>
                        <input type="number" step="0.5" name="number of floors" value="1.5" min="1" max="5" required oninput="document.getElementById('floorVal').textContent = this.value">
                    </div>

                    <div class="form-group">
                        <label>Living Area (sqft)</label>
                        <input type="number" name="living area" value="2570" min="100" required>
                    </div>

                    <div class="form-group">
                        <label>Lot Area (sqft)</label>
                        <input type="number" name="lot area" value="7242" min="100" required>
                    </div>

                    <div class="form-group">
                        <label>Lot Area Renov (sqft)</label>
                        <input type="number" name="lot_area_renov" value="6900" min="100" required>
                    </div>

                    <div class="form-group">
                        <label>Above Area (sqft)</label>
                        <input type="number" name="Area of the house(excluding basement)" value="2170" min="0" required>
                    </div>

                    <div class="form-group">
                        <label>Basement Area (sqft)</label>
                        <input type="number" name="Area of the basement" value="400" min="0" required>
                    </div>
                </div>

                <div class="sub-group-title"><i class="fa-solid fa-gem"></i> Build Grade & View Metrics</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>House Condition (1-5)</label>
                        <div class="select-wrapper">
                            <select name="condition of the house" required>
                                <option value="3" selected>3 - Average Condition</option>
                                <option value="4">4 - Good Condition</option>
                                <option value="5">5 - Pristine / Excellent</option>
                                <option value="2">2 - Fair Condition</option>
                                <option value="1">1 - Poor Condition</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>House Grade (1-13)</label>
                        <input type="number" name="grade of the house" value="8" min="1" max="13" required>
                    </div>

                    <div class="form-group">
                        <label>Waterfront Present</label>
                        <div class="select-wrapper">
                            <select name="waterfront present" required>
                                <option value="0" selected>No Waterfront</option>
                                <option value="1">Waterfront View</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Number of Views (0-4)</label>
                        <input type="number" name="number of views" value="0" min="0" max="4" required>
                    </div>
                </div>

                <div class="sub-group-title"><i class="fa-solid fa-location-dot"></i> Era & Location Attributes</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Built Year</label>
                        <input type="number" name="Built Year" value="1951" min="1800" max="2026" required>
                    </div>

                    <div class="form-group">
                        <label>Renovation Year</label>
                        <input type="number" name="Renovation Year" value="1991" min="0" max="2026" required>
                    </div>

                    <div class="form-group">
                        <label>Nearby Schools</label>
                        <input type="number" name="Number of schools nearby" value="3" min="0" max="20" required>
                    </div>

                    <div class="form-group">
                        <label>Airport Dist (km)</label>
                        <input type="number" step="0.1" name="Distance from the airport" value="12.5" min="0" required>
                    </div>
                </div>

                <button type="submit" class="btn-predict" id="submitBtn">
                    <span class="btn-text">Generate Valuation</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </form>
        </div>

        <!-- Right: Sticky Valuation Sidebar -->
        <div class="sticky-sidebar" id="outputSection">
            <div class="glass-card">
                <div class="section-header">
                    <span class="section-title"><i class="fa-solid fa-chart-line"></i> Market Estimate</span>
                </div>

                <div class="valuation-card">
                    <div class="val-tag">Estimated Asset Value</div>
                    <div class="val-price" id="priceDisplay">$0.00</div>
                    <p style="font-size: 0.8rem; color: var(--text-sub);">Calculated using ordinary least squares (OLS) linear model weight vector.</p>
                </div>

                <div class="feature-badge-grid">
                    <div class="f-badge">
                        <div class="f-badge-title">Primary Area</div>
                        <div class="f-badge-val" id="badgeLiving">2,570 sqft</div>
                    </div>
                    <div class="f-badge">
                        <div class="f-badge-title">Build Age</div>
                        <div class="f-badge-val" id="badgeAge">75 Yrs</div>
                    </div>
                </div>
            </div>
        </div>

    </div>
</div>

<footer>
    &copy; 2026 ValuatX AI Engine &bull; Linear Regression WSGI Container
</footer>

<script>
    document.getElementById('valuationForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const form = e.target;
        const submitBtn = document.getElementById('submitBtn');
        const spinner = document.getElementById('btnSpinner');
        const btnText = submitBtn.querySelector('.btn-text');
        const priceDisplay = document.getElementById('priceDisplay');
        const outputSection = document.getElementById('outputSection');
        
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        btnText.textContent = 'Computing Regression Matrix...';
        
        const livingArea = form.elements['living area'].value;
        const builtYear = form.elements['Built Year'].value;
        document.getElementById('badgeLiving').textContent = Number(livingArea).toLocaleString() + ' sqft';
        document.getElementById('badgeAge').textContent = (2026 - Number(builtYear)) + ' Yrs';
        
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server Error (${response.status}): ${errText.substring(0, 80)}`);
            }
            
            const data = await response.json();
            
            setTimeout(() => {
                if (data.status === 'success') {
                    let price = Number(data.prediction);
                    priceDisplay.textContent = '$' + price.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    });
                    
                    // Smoothly auto-scroll on smaller screens so you don't have to scroll up
                    if (window.innerWidth <= 1024) {
                        outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } else {
                    priceDisplay.textContent = 'Error: ' + data.message;
                }
                
                submitBtn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Generate Valuation';
            }, 400);

        } catch (error) {
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Generate Valuation';
            priceDisplay.textContent = error.message;
        }
    });
</script>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Linear Regression model file was not loaded on the server.'}), 500
        
    try:
        data_dict = {
            'number of bedrooms': [float(request.form['number of bedrooms'])],
            'number of bathrooms': [float(request.form['number of bathrooms'])],
            'living area': [float(request.form['living area'])],
            'lot area': [float(request.form['lot area'])],
            'number of floors': [float(request.form['number of floors'])],
            'waterfront present': [float(request.form['waterfront present'])],
            'number of views': [float(request.form['number of views'])],
            'condition of the house': [float(request.form['condition of the house'])],
            'grade of the house': [float(request.form['grade of the house'])],
            'Area of the house(excluding basement)': [float(request.form['Area of the house(excluding basement)'])],
            'Area of the basement': [float(request.form['Area of the basement'])],
            'Built Year': [float(request.form['Built Year'])],
            'Renovation Year': [float(request.form['Renovation Year'])],
            'lot_area_renov': [float(request.form['lot_area_renov'])],
            'Number of schools nearby': [float(request.form['Number of schools nearby'])],
            'Distance from the airport': [float(request.form['Distance from the airport'])]
        }
        
        features_df = pd.DataFrame(data_dict)
        
        if scaler is not None:
            input_data = scaler.transform(features_df)
        else:
            input_data = features_df
        
        prediction_val = float(model.predict(input_data)[0])
        prediction_val = max(0.0, prediction_val)
        
        return jsonify({
            'status': 'success',
            'prediction': round(prediction_val, 2)
        })
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
