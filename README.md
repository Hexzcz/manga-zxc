# Manga-ZXC

An automated manga translation pipeline with speech bubble detection, OCR, inpainting, and localization features.

## Setup Instructions

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Python 3.10+**
* **Node.js** (LTS recommended)
* **PowerShell** (for Windows startup script)

---

### 2. Backend Setup
1. Open your terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   * **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
4. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up your API Keys:
   * Copy `.env.example` to `.env`
   * Open `.env` and fill in your `DEEPSEEK_API_KEY` or `GEMINI_API_KEY`.

---

### 3. Frontend Setup
1. Navigate to the `frontend` folder:
   ```bash
   cd ../frontend
   ```
2. Install the web dependencies:
   ```bash
   npm install
   ```

---

### 4. Running the Application
To start both servers and launch the application in your browser:
1. Navigate back to the project root directory.
2. Run the startup script:
   * **Windows**:
     ```powershell
     .\run.ps1
     ```
