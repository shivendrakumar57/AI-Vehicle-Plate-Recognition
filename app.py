import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import re, cv2, sqlite3, secrets, time, gc
from datetime import datetime
from pathlib import Path
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

try:
    import easyocr
except Exception:
    easyocr = None

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "static" / "uploads"
RESULT_DIR = APP_DIR / "static" / "results"
DB_PATH = APP_DIR / "anpr_history.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_IMAGES = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}
_reader = None

# UP district list plus commonly used registration prefixes.
UP_RTO = {
    "UP11": "Saharanpur", "UP12": "Muzaffarnagar", "UP13": "Bulandshahr",
    "UP14": "Ghaziabad", "UP15": "Meerut", "UP16": "Gautam Buddha Nagar (Noida)",
    "UP17": "Baghpat", "UP19": "Shamli", "UP20": "Bijnor", "UP21": "Moradabad",
    "UP22": "Rampur", "UP23": "Amroha", "UP24": "Badaun", "UP25": "Bareilly",
    "UP26": "Pilibhit", "UP27": "Shahjahanpur", "UP30": "Hardoi", "UP31": "Lakhimpur Kheri",
    "UP32": "Lucknow", "UP32A": "Trans Gomti, Lucknow (sub-office/reference)",
    "UP33": "Raebareli", "UP34": "Sitapur", "UP35": "Unnao", "UP36": "Amethi/Sultanpur region (verify)",
    "UP37": "Hapur", "UP38": "Sambhal", "UP40": "Bahraich", "UP41": "Barabanki",
    "UP42": "Ayodhya", "UP43": "Gonda", "UP44": "Sultanpur", "UP45": "Ambedkar Nagar",
    "UP46": "Shravasti", "UP47": "Balrampur", "UP50": "Azamgarh", "UP51": "Basti",
    "UP52": "Deoria", "UP53": "Gorakhpur", "UP54": "Mau", "UP55": "Siddharthnagar",
    "UP56": "Mahrajganj", "UP57": "Padrauna/Kushinagar region (verify)", "UP58": "Sant Kabir Nagar",
    "UP60": "Ballia", "UP61": "Ghazipur", "UP62": "Jaunpur", "UP63": "Mirzapur",
    "UP64": "Sonbhadra", "UP65": "Varanasi", "UP66": "Bhadohi", "UP67": "Chandauli",
    "UP70": "Prayagraj", "UP71": "Fatehpur", "UP72": "Pratapgarh", "UP73": "Kaushambi",
    "UP74": "Kannauj", "UP75": "Etawah", "UP76": "Farrukhabad", "UP77": "Kanpur Dehat",
    "UP78": "Kanpur Nagar", "UP79": "Auraiya", "UP80": "Agra", "UP81": "Aligarh",
    "UP82": "Etah", "UP83": "Firozabad", "UP84": "Mainpuri", "UP85": "Mathura",
    "UP86": "Hathras", "UP87": "Kasganj", "UP90": "Banda", "UP91": "Hamirpur",
    "UP92": "Jalaun", "UP93": "Jhansi", "UP94": "Lalitpur", "UP95": "Mahoba",
    "UP96": "Chitrakoot",
}
UP_DISTRICTS = [
"Agra","Aligarh","Ambedkar Nagar","Amethi","Amroha","Auraiya","Ayodhya","Azamgarh",
"Baghpat","Bahraich","Ballia","Balrampur","Banda","Barabanki","Bareilly","Basti",
"Bhadohi","Bijnor","Budaun","Bulandshahr","Chandauli","Chitrakoot","Deoria","Etah",
"Etawah","Farrukhabad","Fatehpur","Firozabad","Gautam Buddha Nagar","Ghaziabad","Ghazipur",
"Gonda","Gorakhpur","Hamirpur","Hapur","Hardoi","Hathras","Jalaun","Jaunpur","Jhansi",
"Kannauj","Kanpur Dehat","Kanpur Nagar","Kasganj","Kaushambi","Kheri (Lakhimpur Kheri)",
"Kushinagar","Lalitpur","Lucknow","Maharajganj","Mahoba","Mainpuri","Mathura","Mau",
"Meerut","Mirzapur","Moradabad","Muzaffarnagar","Pilibhit","Pratapgarh","Prayagraj",
"Raebareli","Rampur","Saharanpur","Sambhal","Sant Kabir Nagar","Shahjahanpur","Shamli",
"Shravasti","Siddharthnagar","Sitapur","Sonbhadra","Sultanpur","Unnao","Varanasi"
]

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            rto_code TEXT,
            district TEXT,
            confidence REAL,
            source TEXT,
            image_path TEXT,
            created_at TEXT NOT NULL
        )""")

def get_reader():
    global _reader
    if _reader is None and easyocr is not None:
        try:
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            _reader = None
    return _reader

def plate_region_candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 40, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    candidates = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < 12 or bw < 45:
            continue
        ratio = bw / max(bh, 1)
        area_ratio = (bw * bh) / max(w * h, 1)
        if 2.0 <= ratio <= 6.8 and 0.0005 <= area_ratio <= 0.12:
            candidates.append((x, y, bw, bh))
        if len(candidates) >= 8:
            break
    return candidates

def clean_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return text[:15]

def identify_rto(plate):
    compact = re.sub(r"[^A-Z0-9]", "", (plate or "").upper())
    m = re.match(r"^(UP\d{2})", compact)
    if not m:
        return "", "Not identified / non-UP"
    code = m.group(1)
    return code, UP_RTO.get(code, "UP code not mapped in this reference list")

def recognize_crop(crop):
    reader = get_reader()
    if reader is None or crop.size == 0:
        return "", 0.0
    try:
        ch, cw = crop.shape[:2]
        if max(ch, cw) > 300:
            scale = 300 / float(max(ch, cw))
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        
        enlarged = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if len(enlarged.shape) == 3 else enlarged
        results = reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not results:
            return "", 0.0
        results.sort(key=lambda x: float(x[2]), reverse=True)
        return clean_plate(results[0][1]), float(results[0][2])
    except Exception:
        return "", 0.0

def process_frame(image, source="upload", save=True):
    if image is None or image.size == 0:
        return image, []
    
    ih, iw = image.shape[:2]
    if max(ih, iw) > 800:
        scale = 800 / float(max(ih, iw))
        image = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

    annotated = image.copy()
    candidates = plate_region_candidates(image)
    detections = []
    for x, y, w, h in candidates:
        pad_x, pad_y = max(3, int(w * .04)), max(3, int(h * .12))
        x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
        x2, y2 = min(image.shape[1], x+w+pad_x), min(image.shape[0], y+h+pad_y)
        crop = image[y1:y2, x1:x2]
        text, conf = recognize_crop(crop)
        if len(text) < 5:
            continue
        if any(d["plate"] == text for d in detections):
            continue
        code, district = identify_rto(text)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (30,220,120), 2)
        label = text if text else "Plate candidate"
        cv2.putText(annotated, label, (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .65, (30,220,120), 2)
        detections.append({"plate": text, "confidence": round(conf, 3), "rto_code": code,
                            "district": district, "bbox": [x1,y1,x2,y2]})
        if save:
            crop_name = f"plate_{int(time.time()*1000)}_{secrets.token_hex(3)}.jpg"
            crop_path = RESULT_DIR / crop_name
            cv2.imwrite(str(crop_path), crop)
            try:
                with db() as con:
                    con.execute("""INSERT INTO detections(plate,rto_code,district,confidence,source,image_path,created_at)
                        VALUES(?,?,?,?,?,?,?)""",
                        (text, code, district, float(conf), source, f"results/{crop_name}",
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            except Exception:
                pass
    
    gc.collect()
    return annotated, detections

def allowed(filename, kinds):
    return "." in filename and filename.rsplit(".",1)[1].lower() in kinds

@app.route("/")
def index():
    try:
        with db() as con:
            total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            unique = con.execute("SELECT COUNT(DISTINCT plate) FROM detections").fetchone()[0]
            recent = con.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 6").fetchall()
    except Exception:
        total, unique, recent = 0, 0, []
    return render_template("index.html", total=total, unique=unique, recent=recent, districts=UP_DISTRICTS)

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Pehle image ya video file select karein.", "error")
        return redirect(url_for("index"))
    name = secure_filename(f.filename)
    ext = name.rsplit(".",1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_IMAGES | ALLOWED_VIDEOS:
        flash("Supported files: JPG, PNG, WEBP, BMP, MP4, AVI, MOV, MKV, WEBM.", "error")
        return redirect(url_for("index"))
    stored = f"{secrets.token_hex(6)}_{name}"
    path = UPLOAD_DIR / stored
    f.save(path)
    all_detections = []
    result_url = None
    try:
        if ext in ALLOWED_IMAGES:
            image = cv2.imread(str(path))
            if image is None:
                flash("Image read nahi ho paayi.", "error")
                return redirect(url_for("index"))
            annotated, all_detections = process_frame(image, source="image upload")
            out_name = "annotated_" + secrets.token_hex(5) + ".jpg"
            cv2.imwrite(str(RESULT_DIR / out_name), annotated)
            result_url = url_for("static", filename=f"results/{out_name}")
        else:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            
            if width > 640:
                scale = 640 / float(width)
                width = 640
                height = int(height * scale)

            out_name = "video_" + secrets.token_hex(5) + ".mp4"
            out_path = RESULT_DIR / out_name
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 15), (width, height))
            frame_idx = 0
            seen = set()
            while cap.isOpened() and frame_idx < 150:
                ok, frame = cap.read()
                if not ok:
                    break
                if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640):
                    frame = cv2.resize(frame, (width, height))
                
                if frame_idx % max(1, int(fps/2)) == 0:
                    annotated, dets = process_frame(frame, source="video upload")
                    for d in dets:
                        if d["plate"] not in seen:
                            seen.add(d["plate"])
                            all_detections.append(d)
                else:
                    annotated = frame
                writer.write(annotated)
                frame_idx += 1
            cap.release()
            writer.release()
            result_url = url_for("static", filename=f"results/{out_name}")
            if not all_detections:
                flash("Video process ho gaya, lekin readable plate detect nahi hui.", "warning")
    except Exception:
        flash("Processing ke dauran error aa gaya.", "error")
        return redirect(url_for("index"))
    finally:
        gc.collect()

    return render_template("result.html", detections=all_detections, result_url=result_url, filename=name)

@app.route("/api/live_frame", methods=["POST"])
def live_frame():
    f = request.files.get("frame")
    if not f:
        return jsonify({"error":"Frame missing"}), 400
    raw = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error":"Invalid frame"}), 400
    annotated, detections = process_frame(image, source="live webcam")
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    import base64
    return jsonify({"image":"data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
                    "detections":detections})

@app.route("/history")
def history():
    q = request.args.get("q", "").strip().upper()
    try:
        with db() as con:
            if q:
                rows = con.execute("SELECT * FROM detections WHERE plate LIKE ? OR district LIKE ? ORDER BY id DESC LIMIT 500",
                                   (f"%{q}%", f"%{q}%")).fetchall()
            else:
                rows = con.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 500").fetchall()
    except Exception:
        rows = []
    return render_template("history.html", rows=rows, q=q)

@app.route("/districts")
def districts():
    return render_template("districts.html", mapping=sorted(UP_RTO.items()), districts=UP_DISTRICTS)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.errorhandler(413)
def too_large(_):
    flash("File 100 MB se chhoti honi chahiye.", "error")
    return redirect(url_for("index"))

init_db()
# Server start hote hi model load kar lo taaki request ke waqt timeout na ho
try:
    print("Pre-loading EasyOCR reader...")
    get_reader()
    print("EasyOCR loaded successfully!")
except Exception as e:
    print("Error loading EasyOCR on startup:", e)
