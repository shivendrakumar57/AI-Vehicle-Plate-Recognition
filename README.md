# PlateVision AI — Vehicle Plate Recognition (ANPR)

## Features
- Flask web dashboard with HTML/CSS
- Image upload and video upload
- Browser webcam snapshots sent to the local Flask app
- OpenCV contour-based plate-candidate detection
- EasyOCR text extraction
- SQLite detection history
- Uttar Pradesh district list and common RTO-prefix reference table

## Important accuracy limitation
This project is a working educational prototype, but the built-in candidate detector is a classical OpenCV heuristic, not a trained plate-specific YOLO model. It may miss plates or identify non-plate rectangles. For a robust ANPR project, train/use a license-plate YOLO model and replace `plate_region_candidates()` in `app.py`. The OCR confidence is not a guarantee that the text is correct. RTO mappings are reference-only and should be checked against official transport data.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- Internet connection for initial Python packages and EasyOCR model download
- Webcam optional

## Run in VS Code (Windows)
1. Extract this ZIP.
2. Open the `AI_Vehicle_Plate_Recognition` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   python app.py
   ```
8. Open `http://127.0.0.1:5000` in Chrome.
9. For live camera, click Start Camera and allow permission. Browser camera access works on localhost; deployed sites need HTTPS.
10. First EasyOCR use may download model files and can take time.

## Database
`anpr_history.db` is created automatically. To reset history, stop the app and delete this database file.

## Privacy
Use only footage/images you have permission to process. Store and share plate images responsibly. This prototype does not query official vehicle owner records and must not be used as an authoritative identity/ownership system.
