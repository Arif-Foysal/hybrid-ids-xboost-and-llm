### **Phase 1: Data Curation & Preprocessing (The Foundation)**

**Goal:** Create a mathematically rigorous dataset that reviewers cannot question.

- **Step 1.1: Data Acquisition:** Download the CSV version of UNSW-NB15 (Train/Test splits)(already done).
    
- **Step 1.2: Sanity Check:** Remove infinite values, drop ID columns (`id`), and handle missing values in the `service` column (replace `-` with `None`).
    
- **Step 1.3: Encoding Strategy (Crucial):**
    
    - Use **Label Encoding** for high-cardinality features (like `source_ip` if kept, though usually dropped).
        
    - Use **One-Hot Encoding** for small categorical features (`state`, `proto`) if you want to test Deep Learning later, but for XGBoost, Label Encoding is sufficient and faster.
        
- **Step 1.4: Feature Selection (Research Value):**
    
    - Don't just use all features. Run a correlation matrix. Drop features that are 99% correlated with others to reduce "noise" and improve speed.
        
- **📝 Paper Deliverable:** "Data Preprocessing" section describing exactly how you handled the dirty data.
    

### **Phase 2: The Detection Engine (The Performance Benchmark)**

**Goal:** Prove your lightweight model is as accurate as heavy "black box" models.

- **Step 2.1: Baseline Training:** Train a default XGBoost classifier.
    
- **Step 2.2: Hyperparameter Optimization (Research Grade):**
    
    - Do not just guess parameters. Use **Optuna** or **GridSearchCV** to find the mathematical best `max_depth`, `learning_rate`, and `n_estimators`.
        
    - _Why?_ Journals require proof that you tuned your model.
        
- **Step 2.3: Comparative Training:**
    
    - Train a **Random Forest** and a simple **Deep Neural Network (DNN)** on the same data.
        
    - _Why?_ You need these for your comparison table to show XGBoost is the best balance of speed vs. accuracy.
        
- **📝 Paper Deliverable:** A "Performance Table" showing Accuracy, Precision, Recall, and F1-Score for XGBoost vs. RF vs. DNN.
    

### **Phase 3: The Interpretability Layer (The "Why")**

**Goal:** Extract mathematical justification for every alert.

- **Step 3.1: Global Interpretability:** Use `shap.TreeExplainer` to generate a "Summary Plot." This shows which features (e.g., `sload`, `dbytes`) drive attacks globally.
    
- **Step 3.2: Local Interpretability:** Write a function that takes a _single_ attack row and returns the top 5 features contributing to that specific prediction.
    
- **📝 Paper Deliverable:** A "SHAP Summary Plot" (Figure 1 in your results) and a case study of a specific attack (e.g., "Analysis of a Fuzzer Attack").
    

### **Phase 4: The Generative Narrative (The Novelty)**

**Goal:** Convert math (SHAP) into language (English) entirely offline.

- **Step 4.1: Quantization Setup:** Load **Llama-3-8B-Instruct** using `bitsandbytes` (4-bit quantization) on your GPU. Verify it consumes < 8GB VRAM.
    
- **Step 4.2: The "Bridge" Prompt Engineering:**
    
    - Develop a dynamic prompt template:
        
    - _Template:_ "System: You are a security analyst. Input: Attack Type = {Y}, Key Features = {X1, X2, X3}. Task: Explain why this traffic is malicious."
        
- **Step 4.3: Batch Inference:** Run this pipeline on 50 different attack samples.
    
- **📝 Paper Deliverable:** A table comparing "Raw Log Data" vs. "Your Generated Report."
    

### **Phase 5: Edge Feasibility Analysis (The "Edge" Contribution)**

**Goal:** Prove this can run on a laptop/edge device (the "Gap" you identified).

- **Step 5.1: Latency Testing:**
    
    - Measure time to detect (XGBoost inference time).
        
    - Measure time to explain (SHAP calculation time).
        
    - Measure time to generate (Llama-3 generation time).
        
- **Step 5.2: Resource Monitoring:**
    
    - Use Python libraries (`psutil`, `nvidia-smi`) to log CPU, RAM, and GPU VRAM usage during operation.
        
- **📝 Paper Deliverable:** A "Resource Utilization" table. This is your key argument for "Edge Deployment."