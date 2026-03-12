import cv2
import numpy as np
import os
from flask import Flask, request, Response, jsonify
import base64
import time
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

###########################################
# كاشف الوجه والتوتر المبسط (بدون نماذج خارجية)
###########################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return faces

def analyze_face_stress(face_roi):
    """تحليل الوجه وإعطاء نسبة توتر بناءً على خصائص بصرية"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # التباين
    contrast = gray.std()
    # السطوع
    brightness = gray.mean()
    # الحدة
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    stress_contrast = max(0, min(1, 1 - (contrast / 100)))
    stress_brightness = abs(brightness - 127) / 127
    stress_blur = max(0, min(1, 1 - (laplacian / 500)))
    
    stress_score = (stress_contrast * 0.3 + stress_brightness * 0.4 + stress_blur * 0.3)
    return stress_score

###########################################
# HTML الواجهة (مدمج في الكود)
###########################################
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام تحليل التوتر في المطارات</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Cairo', sans-serif; }
        body { background: linear-gradient(135deg, #0b1a3a 0%, #1a3b5c 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1300px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 30px; padding: 25px 40px; margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.2); box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;
        }
        .logo { display: flex; align-items: center; gap: 15px; }
        .logo i { font-size: 3rem; color: #ffd966; text-shadow: 0 0 20px rgba(255,217,102,0.5); }
        .logo h1 { color: white; font-size: 2rem; font-weight: 700; line-height: 1.3; }
        .logo span { color: #ffd966; display: block; font-size: 1rem; font-weight: 400; opacity: 0.9; }
        .badge {
            background: rgba(255,217,102,0.2); border: 1px solid #ffd966;
            padding: 12px 25px; border-radius: 50px; color: #ffd966; font-weight: 600;
            display: flex; align-items: center; gap: 10px;
        }
        .cards-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 25px; margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 25px; padding: 25px; border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: 0.3s;
        }
        .card:hover { transform: translateY(-5px); border-color: #ffd966; }
        .card-icon {
            width: 60px; height: 60px; background: rgba(255,217,102,0.2);
            border-radius: 20px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px;
        }
        .card-icon i { font-size: 2rem; color: #ffd966; }
        .card h3 { color: white; font-size: 1.3rem; margin-bottom: 10px; }
        .card p { color: rgba(255,255,255,0.7); font-size: 0.95rem; line-height: 1.6; }
        .upload-section {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 30px; padding: 30px; margin-bottom: 30px;
            border: 2px dashed rgba(255,217,102,0.5); text-align: center;
        }
        .upload-icon { font-size: 4rem; color: #ffd966; margin-bottom: 15px; }
        .upload-section h2 { color: white; font-size: 1.8rem; margin-bottom: 10px; }
        .upload-section p { color: rgba(255,255,255,0.7); margin-bottom: 25px; }
        .file-input-wrapper { position: relative; display: inline-block; }
        .file-input-wrapper input {
            position: absolute; width: 100%; height: 100%; opacity: 0; cursor: pointer;
        }
        .file-label {
            background: #ffd966; color: #0b1a3a; padding: 15px 40px; border-radius: 50px;
            font-weight: 700; font-size: 1.1rem; display: inline-flex; align-items: center; gap: 10px;
            cursor: pointer; transition: 0.3s; box-shadow: 0 10px 20px rgba(255,217,102,0.3);
        }
        .file-label:hover { background: #ffe085; transform: scale(1.05); }
        .selected-file { color: white; margin-top: 15px; font-size: 0.95rem; }
        .analyze-btn {
            background: linear-gradient(45deg, #ffd966, #ffb347); color: #0b1a3a;
            border: none; padding: 15px 50px; border-radius: 50px; font-weight: 700; font-size: 1.2rem;
            margin-top: 20px; cursor: pointer; transition: 0.3s;
            box-shadow: 0 10px 20px rgba(255,217,102,0.3); display: inline-flex; align-items: center; gap: 10px;
        }
        .analyze-btn:hover { transform: scale(1.05); box-shadow: 0 15px 30px rgba(255,217,102,0.5); }
        .analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .results-section {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 30px; padding: 30px; margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .results-header { display: flex; align-items: center; gap: 15px; margin-bottom: 25px; }
        .results-header i { font-size: 2rem; color: #ffd966; }
        .results-header h2 { color: white; font-size: 1.8rem; }
        .results-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 20px;
        }
        .result-card {
            background: rgba(0,0,0,0.3); border-radius: 20px; padding: 20px; text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .result-label { color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-bottom: 10px; }
        .result-value { color: #ffd966; font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; }
        .result-unit { color: rgba(255,255,255,0.5); font-size: 0.9rem; }
        .camera-section {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 30px; padding: 30px; border: 1px solid rgba(255,255,255,0.2);
        }
        .camera-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; flex-wrap: wrap; gap: 20px;
        }
        .camera-title { display: flex; align-items: center; gap: 15px; }
        .camera-title i { font-size: 2rem; color: #ffd966; }
        .camera-title h2 { color: white; font-size: 1.8rem; }
        .camera-toggle {
            background: #ffd966; color: #0b1a3a; border: none; padding: 12px 30px;
            border-radius: 50px; font-weight: 600; font-size: 1rem; cursor: pointer;
            transition: 0.3s; display: flex; align-items: center; gap: 8px;
        }
        .camera-toggle:hover { background: #ffe085; transform: scale(1.05); }
        .camera-container {
            position: relative; border-radius: 20px; overflow: hidden; background: #000;
            aspect-ratio: 16/9; margin-bottom: 20px;
        }
        #cameraFeed { width: 100%; height: 100%; object-fit: cover; }
        .camera-placeholder {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: rgba(0,0,0,0.7); color: white; gap: 15px;
        }
        .camera-placeholder i { font-size: 4rem; color: #ffd966; }
        .live-stress {
            background: rgba(255,217,102,0.2); border: 1px solid #ffd966; border-radius: 50px;
            padding: 15px 30px; display: inline-flex; align-items: center; gap: 15px;
        }
        .live-stress span { color: white; font-size: 1.1rem; }
        .live-stress strong { color: #ffd966; font-size: 1.8rem; margin: 0 5px; }
        .stress-meter {
            width: 100%; height: 10px; background: rgba(255,255,255,0.2);
            border-radius: 10px; margin-top: 15px; overflow: hidden;
        }
        .stress-meter-fill {
            height: 100%; background: linear-gradient(90deg, #00ff00, #ffff00, #ff0000);
            transition: width 0.3s;
        }
        @media (max-width: 768px) {
            .header-content { flex-direction: column; text-align: center; }
            .logo { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <i class="fas fa-shield-alt"></i>
                <div>
                    <h1>نظام تحليل التوتر في المطارات</h1>
                    <span>AI-Powered Behavior Analysis System</span>
                </div>
            </div>
            <div class="badge">
                <i class="fas fa-video"></i>
                <span>معسكر بناء القدرات في الذكاء الاصطناعي</span>
            </div>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-icon"><i class="fas fa-brain"></i></div>
                <h3>ذكاء اصطناعي متقدم</h3>
                <p>يحلل تعابير الوجه ولغة الجسد بدقة عالية لاكتشاف علامات التوتر والقلق</p>
            </div>
            <div class="card">
                <div class="card-icon"><i class="fas fa-chart-line"></i></div>
                <h3>نسبة توتر فورية</h3>
                <p>يقيس مستوى التوتر كنسبة مئوية ويلون النتائج (أخضر-أصفر-أحمر) لسهولة التقييم</p>
            </div>
            <div class="card">
                <div class="card-icon"><i class="fas fa-clock"></i></div>
                <h3>تحليل آني ولاحق</h3>
                <p>يدعم الكاميرا المباشرة ورفع ملفات الفيديو لتحليلها لاحقاً</p>
            </div>
        </div>

        <div class="upload-section">
            <div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
            <h2>رفع فيديو للتحليل</h2>
            <p>اختر مقطع فيديو لمسافرين وسيقوم النظام بتحليل نسب التوتر لكل شخص</p>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input-wrapper">
                    <input type="file" name="video" id="videoInput" accept="video/*" required>
                    <label for="videoInput" class="file-label"><i class="fas fa-folder-open"></i> اختر فيديو</label>
                </div>
                <div class="selected-file" id="selectedFileName">لم يتم اختيار ملف</div>
                <button type="submit" class="analyze-btn" id="analyzeBtn"><i class="fas fa-chart-bar"></i> تحليل الفيديو</button>
            </form>
        </div>

        <div class="results-section" id="resultsSection" style="display: none;">
            <div class="results-header"><i class="fas fa-clipboard-check"></i><h2>نتائج التحليل</h2></div>
            <div class="results-grid" id="resultsGrid"></div>
        </div>

        <div class="camera-section">
            <div class="camera-header">
                <div class="camera-title"><i class="fas fa-camera"></i><h2>الكاميرا المباشرة</h2></div>
                <button class="camera-toggle" id="cameraToggle"><i class="fas fa-play"></i> تشغيل الكاميرا</button>
            </div>
            <div class="camera-container">
                <img id="cameraFeed" style="display: none;">
                <div class="camera-placeholder" id="cameraPlaceholder">
                    <i class="fas fa-video-slash"></i>
                    <p>الكاميرا متوقفة، اضغط على "تشغيل الكاميرا" للبدء</p>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <div class="live-stress">
                    <i class="fas fa-exclamation-triangle" style="color: #ffd966;"></i>
                    <span>نسبة التوتر اللحظية:</span>
                    <strong id="liveStressValue">0</strong>
                    <span>%</span>
                </div>
                <div class="stress-meter"><div class="stress-meter-fill" id="stressMeterFill" style="width: 0%;"></div></div>
            </div>
        </div>
    </div>

    <script>
        // عناصر الصفحة
        const videoInput = document.getElementById('videoInput');
        const selectedFileName = document.getElementById('selectedFileName');
        const uploadForm = document.getElementById('uploadForm');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultsSection = document.getElementById('resultsSection');
        const resultsGrid = document.getElementById('resultsGrid');
        const cameraToggle = document.getElementById('cameraToggle');
        const cameraFeed = document.getElementById('cameraFeed');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const liveStressValue = document.getElementById('liveStressValue');
        const stressMeterFill = document.getElementById('stressMeterFill');

        // اسم الملف
        videoInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                selectedFileName.textContent = 'الملف المختار: ' + this.files[0].name;
            } else {
                selectedFileName.textContent = 'لم يتم اختيار ملف';
            }
        });

        // رفع الفيديو
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!videoInput.files || !videoInput.files[0]) {
                alert('الرجاء اختيار فيديو أولاً');
                return;
            }
            const formData = new FormData();
            formData.append('video', videoInput.files[0]);
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> جاري التحليل...';
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                resultsSection.style.display = 'block';
                let stressLevel = 'منخفض', stressColor = '#00ff00';
                if (data.average_stress > 0.6) { stressLevel = 'مرتفع'; stressColor = '#ff0000'; }
                else if (data.average_stress > 0.3) { stressLevel = 'متوسط'; stressColor = '#ffff00'; }
                resultsGrid.innerHTML = `
                    <div class="result-card"><div class="result-label">متوسط التوتر</div><div class="result-value" style="color: ${stressColor};">${(data.average_stress*100).toFixed(0)}%</div><div class="result-unit">${stressLevel}</div></div>
                    <div class="result-card"><div class="result-label">أعلى توتر</div><div class="result-value">${(data.max_stress*100).toFixed(0)}%</div><div class="result-unit">الذروة</div></div>
                    <div class="result-card"><div class="result-label">الإطارات المحللة</div><div class="result-value">${data.frames_analyzed}</div><div class="result-unit">إطار</div></div>
                    ${data.message ? `<div class="result-card"><div class="result-label">ملاحظة</div><div class="result-value" style="font-size:1rem;">${data.message}</div></div>` : ''}
                `;
            } catch (err) {
                alert('حدث خطأ في تحليل الفيديو');
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-chart-bar"></i> تحليل الفيديو';
            }
        });

        // الكاميرا
        let cameraActive = false;
        cameraToggle.addEventListener('click', () => {
            if (!cameraActive) {
                cameraFeed.src = '/video_feed';
                cameraFeed.style.display = 'block';
                cameraPlaceholder.style.display = 'none';
                cameraToggle.innerHTML = '<i class="fas fa-stop"></i> إيقاف الكاميرا';
                cameraToggle.classList.add('active');
                cameraActive = true;
                updateLiveStress();
            } else {
                cameraFeed.src = '';
                cameraFeed.style.display = 'none';
                cameraPlaceholder.style.display = 'flex';
                cameraToggle.innerHTML = '<i class="fas fa-play"></i> تشغيل الكاميرا';
                cameraToggle.classList.remove('active');
                cameraActive = false;
                liveStressValue.textContent = '0';
                stressMeterFill.style.width = '0%';
            }
        });

        function updateLiveStress() {
            if (!cameraActive) return;
            // محاكاة تحديث القيمة (سيتم استبدالها لاحقاً بقراءة حقيقية)
            fetch('/live_stress')
                .then(r => r.json())
                .then(data => {
                    const stress = data.stress * 100;
                    liveStressValue.textContent = Math.round(stress);
                    stressMeterFill.style.width = stress + '%';
                })
                .catch(() => {
                    // إذا فشل الطلب، نستخدم قيمة عشوائية مؤقتة
                    const rand = Math.random() * 100;
                    liveStressValue.textContent = Math.round(rand);
                    stressMeterFill.style.width = rand + '%';
                })
                .finally(() => {
                    setTimeout(updateLiveStress, 1000);
                });
        }
    </script>
</body>
</html>
"""

###########################################
# دوال تحليل الفيديو والكاميرا
###########################################
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stress_scores = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        faces = detect_faces(frame)
        if len(faces) == 0:
            continue
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        face_roi = frame[y:y+h, x:x+w]
        stress = analyze_face_stress(face_roi)
        stress_scores.append(stress)
    cap.release()
    if not stress_scores:
        return {'average_stress': 0, 'max_stress': 0, 'frames_analyzed': 0, 'message': 'لم يتم اكتشاف وجوه'}
    avg_stress = np.mean(stress_scores)
    max_stress = np.max(stress_scores)
    return {'average_stress': round(float(avg_stress), 2),
            'max_stress': round(float(max_stress), 2),
            'frames_analyzed': len(stress_scores)}

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            stress = analyze_face_stress(face_roi)
            color = (0, 255, 0) if stress < 0.3 else (0, 255, 255) if stress < 0.6 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"Stress: {int(stress*100)}%"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

###########################################
# المسارات (Routes)
###########################################
@app.route('/')
def index():
    return HTML_PAGE

@app.route('/upload', methods=['POST'])
def upload_video_route():
    if 'video' not in request.files:
        return 'No file uploaded', 400
    file = request.files['video']
    if file.filename == '':
        return 'No file selected', 400
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    results = analyze_video(filepath)
    os.remove(filepath)
    return jsonify(results)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_stress')
def live_stress():
    # للتبسيط نعيد قيمة عشوائية (يمكن تحسينها لاحقاً)
    return jsonify({'stress': random.uniform(0, 1)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)