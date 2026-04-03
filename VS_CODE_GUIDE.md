# 🚀 VS Code Setup and Running Guide

## Step 1: Open Project in VS Code

1. Open VS Code
2. Click **File** → **Open Folder**
3. Navigate to: `C:\Users\Lenovo\OneDrive\Desktop\water`
4. Click **Select Folder**

## Step 2: Open Terminal in VS Code

1. Press `Ctrl + `` (backtick) to open terminal
   - OR click **Terminal** → **New Terminal**
2. Make sure you're in the project directory:
   ```bash
   cd C:\Users\Lenovo\OneDrive\Desktop\water
   ```

## Step 3: Install Dependencies

In the VS Code terminal, run:
```bash
pip install -r requirements.txt
```

Wait for all packages to install.

## Step 4: Run the Project Components

### Option A: Run Everything in Sequence

Run these commands one by one in the VS Code terminal:

#### 1. Generate Data
```bash
python src/generate_data.py
```
**Expected Output**: Data file created at `data/water_demand.csv`

#### 2. Train Models
```bash
python src/model_train.py
```
**Expected Output**: Models saved to `models/` directory

#### 3. Evaluate Models
```bash
python src/evaluate.py
```
**Expected Output**: Evaluation metrics printed to console

#### 4. Generate Visualizations
```bash
python src/visualize.py
```
**Expected Output**: Plots saved to `data/plots/` directory

#### 5. Run Dashboard
```bash
python -m streamlit run src/app.py
```
**Expected Output**: Dashboard opens in browser at `http://localhost:8501`

---

### Option B: Use VS Code Run Configuration

1. Click on any Python file (e.g., `src/generate_data.py`)
2. Click the **Run** button (▶️) in the top right
3. Or press `F5` to run with debugging

---

## Step 5: View Results

### View Data
- Open `data/water_demand.csv` in VS Code
- Right-click → **Open Preview** or use CSV viewer extension

### View Plots
- Open `data/plots/` folder in VS Code
- Right-click any `.png` file → **Open Preview**

### View Models
- Models are saved in `models/` directory as `.pkl` files
- These are binary files (not meant to be opened directly)

---

## Step 6: Run Streamlit Dashboard

1. In VS Code terminal, run:
   ```bash
   python -m streamlit run src/app.py
   ```

2. The terminal will show:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```

3. VS Code will automatically open your browser, OR:
   - Click the link in the terminal
   - Or manually open: `http://localhost:8501`

4. To stop the dashboard:
   - Press `Ctrl + C` in the terminal

---

## 📝 Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python src/generate_data.py

# Train models
python src/model_train.py

# Evaluate models
python src/evaluate.py

# Generate visualizations
python src/visualize.py

# Run dashboard
python -m streamlit run src/app.py
```

---

## 🔧 VS Code Extensions (Optional but Recommended)

1. **Python** - Microsoft (for Python support)
2. **Pylance** - Microsoft (for better code intelligence)
3. **Jupyter** - Microsoft (if you want to use notebooks)
4. **Python Docstring Generator** - For documentation

To install extensions:
1. Click the Extensions icon (📦) in the sidebar
2. Search for the extension name
3. Click **Install**

---

## 🐛 Troubleshooting

### Issue: "python is not recognized"
**Solution**: 
- Make sure Python is installed and in PATH
- Or use: `py src/generate_data.py` instead of `python`

### Issue: "Module not found"
**Solution**: 
- Make sure you're in the project directory
- Run: `pip install -r requirements.txt`

### Issue: "streamlit is not recognized"
**Solution**: 
- Use: `python -m streamlit run src/app.py`
- Not: `streamlit run src/app.py`

### Issue: Dashboard won't open
**Solution**:
- Check if port 8501 is already in use
- Try: `python -m streamlit run src/app.py --server.port 8502`

### Issue: Models not found
**Solution**:
- Run `python src/model_train.py` first
- Make sure models are in `models/` directory

---

## 📂 Project Structure in VS Code

```
water/
├── 📁 data/              # Data files
│   ├── 📁 plots/         # Visualization plots
│   └── 📄 water_demand.csv
├── 📁 models/            # Trained models
│   ├── quantile_models.pkl
│   ├── sarimax_model.pkl
│   └── scaler.pkl
├── 📁 src/               # Source code
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── model_train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── app.py
├── 📄 requirements.txt   # Dependencies
├── 📄 README.md         # Documentation
└── 📄 VS_CODE_GUIDE.md  # This file
```

---

## 🎯 Recommended Workflow

1. **First Time Setup**:
   - Open project in VS Code
   - Install dependencies: `pip install -r requirements.txt`
   - Generate data: `python src/generate_data.py`
   - Train models: `python src/model_train.py`

2. **Daily Use**:
   - Run dashboard: `python -m streamlit run src/app.py`
   - View results in browser
   - Check plots in `data/plots/`

3. **Making Changes**:
   - Edit code in VS Code
   - Run individual scripts to test
   - Use VS Code's built-in terminal for quick testing

---

## 💡 Tips

1. **Use VS Code's Integrated Terminal**: 
   - Press `` Ctrl + ` `` to toggle terminal
   - Split terminal: Click the split icon (⧉)

2. **Run Python Files**:
   - Right-click file → **Run Python File in Terminal**
   - Or use the Run button (▶️) in top right

3. **Debugging**:
   - Set breakpoints by clicking left of line numbers
   - Press `F5` to start debugging
   - Use `F10` to step over, `F11` to step into

4. **Viewing Files**:
   - Double-click files to open
   - Use Ctrl+P to quickly find files
   - Use Ctrl+Shift+F to search across all files

---

## 🚀 Quick Start Checklist

- [ ] Project opened in VS Code
- [ ] Terminal opened (Ctrl + `)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data generated (`python src/generate_data.py`)
- [ ] Models trained (`python src/model_train.py`)
- [ ] Models evaluated (`python src/evaluate.py`)
- [ ] Visualizations generated (`python src/visualize.py`)
- [ ] Dashboard running (`python -m streamlit run src/app.py`)

---

**You're all set! Happy coding! 🎉**






