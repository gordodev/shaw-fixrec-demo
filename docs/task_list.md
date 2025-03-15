## **📌 Development Order **

### **1️⃣ Setup & Core Framework (1-2 hours)**
- [x] `config.py` → Define file paths, FIX message keys, and settings.  
- [x] `models.py` → Create `SecurityMasterEntry`, `FixMessage`, etc. using `dataclasses`.  
- [x] `security_master.py` → Load **Secmaster CSV**, index by `SecurityID/ISIN/CUSIP`.  
- [ ] `fix_parser.py` → Read **FIX logs**, extract `Symbol`, `Price`, `Quantity`, etc.  

### **2️⃣ Reconciliation Logic (3-4 hours)**
- [ ] `analyzer.py` → Compare parsed FIX messages against Secmaster data.  
- [ ] `reporter.py` → Generate **CSV reports** showing mismatches & financial impact.  
- [ ] `main.py` → Wire everything together, call the **parser → analyzer → reporter**.  

### **3️⃣ Logging, Error Handling, & Security (1-2 hours)**
- [ ] `logging_config.py` → Setup **structured logs** for debugging.  
- [ ] `utils.py` → Add helper functions (e.g., timestamp conversion, validation checks).  
- [ ] **Enhance `fix_parser.py` & `security_master.py`** with **error handling** for bad data.  

### **4️⃣ Testing & Documentation (3-4 hours)**
- [ ] `tests/test_fix_parser.py` → Test if FIX logs parse correctly.  
- [ ] `tests/test_security_master.py` → Test Secmaster lookups.  
- [ ] `tests/test_analyzer.py` → Test reconciliation logic.  
- [ ] `README.md` → Write installation steps, usage guide, and example outputs.  
- [ ] `requirements.txt` → List dependencies (`pandas`, `pytest`, etc.).  

### **5️⃣ Cleanup & Final Checks (1 hour)**
- [ ] `.gitignore` → Exclude **logs, compiled files, and local configs**.  
- [ ] `sample_data/` → Ensure example FIX logs & Secmaster CSV are clean.  
- [ ] **Run final tests** (`pytest tests/`) and fix any issues.  
- [ ] **Code review & performance tuning** (optimize loops, add logging).  
