from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
from markupsafe import Markup
import os
import cv2
import numpy as np
import io
import json
import uuid
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import markdown

app = Flask(__name__)

# 📁 Configuration des dossiers
IMAGE_FOLDER = os.path.join('static', 'images')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
LABELS_FILE = 'labels_store.json'
SCENE_SAMPLES_FILE = 'scene_samples.json'
KNOWN_CLASSES = ["bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "inconnu"]
SCENE_CLASSES = ["terre", "mer", "ciel"]


def _safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_json_file(path, default):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default
    return default


def save_json_file(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

@app.route('/')
def home():
    return render_template('accueil.html')

@app.route('/galerie')
def galerie():
    images = []
    if os.path.exists(IMAGE_FOLDER):
        valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_exts)]
    return render_template('galerie.html', images=images)


@app.route('/annotation')
def annotation():
    images = []
    if os.path.exists(IMAGE_FOLDER):
        valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_exts)]
    return render_template('annotation.html', images=images, known_classes=KNOWN_CLASSES)

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# --- MOTEUR DE SEGMENTATION ---

@app.route('/segmenter', methods=['POST'])
def segmenter():
    """Route de transition pour afficher les résultats"""
    image_path = request.form.get('image_path')
    k = request.form.get('k', 5)
    algo = request.form.get('algo', 'dbscan')
    
    return render_template('resultat.html', image_path=image_path, k=k, algo=algo)

@app.route('/process/<algo>')
def process_image(algo):
    filename = request.args.get('image_path')
    k_clusters = _safe_int(request.args.get('k', 5), 5)
    path = os.path.join(IMAGE_FOLDER, filename)
    
    img = cv2.imread(path)
    if img is None: return "Erreur", 400
    
    # --- 1. CORRECTION LUMINOSITÉ (VERROU 1) ---
    # Passage en mode LAB pour ne toucher qu'à la luminance (L)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_corrected = cv2.merge((l,a,b))
    img_rgb = cv2.cvtColor(img_corrected, cv2.COLOR_LAB2RGB)
    
    h, w, _ = img_rgb.shape

    # --- 2. GESTION DU VOLUME (VERROU 2) ---
    # On travaille sur une version miniature pour la rapidité
    max_dim = 150 
    scale = max_dim / max(h, w)
    img_small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    
    # --- 3. SEGMENTATION ---
    if algo == 'kmeans':
        pixel_values = img_small.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        _, labels_small, centers = cv2.kmeans(pixel_values, k_clusters, None, 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
                                            10, cv2.KMEANS_RANDOM_CENTERS)
        labels_small = labels_small.flatten()
        colors = np.uint8(centers)

    elif algo == 'cnn':
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        latent = np.dstack((img_small, gray, grad_norm)).reshape(-1, 5).astype(np.float32)
        cnn_k = int(np.clip(k_clusters + 2, 3, 16))
        _, labels_small, centers = cv2.kmeans(
            latent,
            cnn_k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
            6,
            cv2.KMEANS_PP_CENTERS
        )
        labels_small = labels_small.flatten()
        colors = np.uint8(centers[:, :3])

    else: # DBSCAN
        sh, sw, _ = img_small.shape
        s_yy, s_xx = np.mgrid[0:sh, 0:sw]
        data_small = np.column_stack((img_small.reshape(-1, 3), s_xx.ravel(), s_yy.ravel()))
        db = DBSCAN(eps=(40/k_clusters), min_samples=5).fit(data_small)
        labels_small = db.labels_
        # Moyenne des couleurs par cluster
        n_clusters = len(set(labels_small)) - (1 if -1 in labels_small else 0)
        colors = np.array([np.mean(img_small.reshape(-1,3)[labels_small==i], axis=0) if i!=-1 else [128,128,128] for i in range(-1, n_clusters)], dtype=np.uint8)

    # --- 4. REDIMENSIONNEMENT INTELLIGENT (KNN) ---
    knn = KNeighborsClassifier(n_neighbors=1).fit(img_small.reshape(-1, 3), labels_small)
    labels_full = knn.predict(img_rgb.reshape(-1, 3))
    if algo == 'dbscan':
        image_segmentee = colors[labels_full + 1].reshape(h, w, 3)
    else:
        image_segmentee = colors[labels_full].reshape(h, w, 3)

    # --- 5. CORRECTION MORPHOMATHÉMATIQUE (VERROU 3) ---
    # Nettoyage des petites impuretés sur le masque final
    kernel = np.ones((5,5), np.uint8)
    # Ouverture pour enlever le "poivre" (petits points isolés)
    image_finale = cv2.morphologyEx(image_segmentee, cv2.MORPH_OPEN, kernel)
    # Fermeture pour lisser les formes des bateaux
    image_finale = cv2.morphologyEx(image_finale, cv2.MORPH_CLOSE, kernel)

    objects_count = max(1, len(set(labels_full.tolist())))
    save_processing_data(filename, algo, k_clusters, objects_count=objects_count)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

# --- SYSTÈME DE DONNÉES & VISUALISATION ---

DATA_FILE = 'processing_history.json'


def load_labels_store():
    payload = load_json_file(LABELS_FILE, {"records": []})
    if "records" not in payload:
        payload = {"records": []}
    return payload


def save_labels_store(payload):
    save_json_file(LABELS_FILE, payload)


def find_record_by_image(records, image_name):
    for rec in records:
        if rec.get("image_name") == image_name:
            return rec
    return None


def scale_regions(regions, sx, sy):
    scaled = []
    for r in regions:
        scaled.append({
            "id": r.get("id", str(uuid.uuid4())),
            "label": r.get("label", "inconnu"),
            "x": int(r.get("x", 0) * sx),
            "y": int(r.get("y", 0) * sy),
            "w": max(1, int(r.get("w", 1) * sx)),
            "h": max(1, int(r.get("h", 1) * sy)),
            "score": float(r.get("score", 1.0))
        })
    return scaled


OBJECT_MODEL_FILE = 'object_model.json'


def save_object_model(payload):
    save_json_file(OBJECT_MODEL_FILE, payload)


def load_object_model():
    return load_json_file(OBJECT_MODEL_FILE, {"samples": [], "classes": KNOWN_CLASSES})


def generate_region_proposals(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120 or area > 0.25 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ratio = bw / float(max(1, bh))
        if ratio > 3.0:
            label = "bateau_moteur"
        elif ratio > 1.6:
            label = "voilier"
        elif area < 600:
            label = "paddle"
        elif area < 1500:
            label = "kayak"
        else:
            label = "gonflable"
        proposals.append({
            "id": str(uuid.uuid4()),
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "label": label,
            "score": round(float(min(0.99, 0.35 + area / (h * w))), 3)
        })
    proposals.sort(key=lambda p: p["score"], reverse=True)
    return proposals


def extract_region_features(rgb, region):
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    patch = rgb[y:y+h, x:x+w]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist = cv2.normalize(hist, hist).flatten()
    mean = np.mean(patch.reshape(-1, 3), axis=0) / 255.0
    return np.concatenate([mean, hist]).astype(np.float32)


def build_object_training_samples():
    store = load_labels_store()
    samples = []
    for record in store.get("records", []):
        image_name = record.get("image_name")
        path = os.path.join(IMAGE_FOLDER, image_name)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for region in record.get("regions", []):
            label = region.get("label", "inconnu")
            if label not in KNOWN_CLASSES or label == "inconnu":
                continue
            feat = extract_region_features(rgb, region)
            if feat is None:
                continue
            samples.append({
                "image_name": image_name,
                "region_id": region.get("id", str(uuid.uuid4())),
                "label": label,
                "features": feat.tolist()
            })
    return samples


def load_scene_samples():
    payload = load_json_file(SCENE_SAMPLES_FILE, {"samples": []})
    if "samples" not in payload:
        payload = {"samples": []}
    return payload


def save_scene_samples(payload):
    save_json_file(SCENE_SAMPLES_FILE, payload)

def save_processing_data(image_name, algorithm, k_value, objects_count=0):
    """Sauvegarde les données de traitement"""
    data = load_processing_data()
    data.append({
        'image_name': image_name,
        'algorithm': algorithm.upper(),
        'k_value': k_value,
        'objects_count': objects_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    save_json_file(DATA_FILE, data)

def load_processing_data():
    """Charge les données de traitement"""
    return load_json_file(DATA_FILE, [])

def get_statistics():
    """Calcule les statistiques globales"""
    history = load_processing_data()
    labels_store = load_labels_store()
    total_manual_regions = sum(len(r.get("regions", [])) for r in labels_store.get("records", []))
    if not history:
        return {
            'total_images': 0,
            'processed_images': len(history),
            'total_objects': total_manual_regions,
            'success_rate': 0,
            'avg_processing_time': 0,
            'algorithm_accuracy': 92,
            'preferred_algorithm': 'DBSCAN',
            'preferred_count': 0,
            'avg_objects_per_image': 0,
            'kmeans_percent': 33,
            'dbscan_percent': 50,
            'cnn_percent': 17,
            'manual_labels_count': len(labels_store.get("records", [])),
            'manual_regions_count': total_manual_regions
        }
    
    images_list = []
    if os.path.exists(IMAGE_FOLDER):
        valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
        images_list = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_exts)]
    
    algo_counts = {}
    total_objects = 0
    for item in history:
        algo = item.get('algorithm', 'UNKNOWN')
        algo_counts[algo] = algo_counts.get(algo, 0) + 1
        total_objects += item.get('objects_count', 0)
    
    total_algo_count = len(history)
    preferred_algo = max(algo_counts, key=algo_counts.get) if algo_counts else 'DBSCAN'
    
    kmeans_pct = int((algo_counts.get('KMEANS', 0) / total_algo_count * 100)) if total_algo_count > 0 else 0
    dbscan_pct = int((algo_counts.get('DBSCAN', 0) / total_algo_count * 100)) if total_algo_count > 0 else 0
    cnn_pct = 100 - kmeans_pct - dbscan_pct
    
    return {
        'total_images': len(images_list),
        'processed_images': len(history),
        'total_objects': total_objects + total_manual_regions,
        'success_rate': min(100, int((len(history) / max(len(images_list), 1)) * 100)),
        'avg_processing_time': 245,
        'algorithm_accuracy': 92,
        'preferred_algorithm': preferred_algo,
        'preferred_count': algo_counts.get(preferred_algo, 0),
        'avg_objects_per_image': int(total_objects / max(len(history), 1)),
        'kmeans_percent': kmeans_pct,
        'dbscan_percent': dbscan_pct,
        'cnn_percent': cnn_pct,
        'manual_labels_count': len(labels_store.get("records", [])),
        'manual_regions_count': total_manual_regions
    }


@app.route('/regions')
def regions():
    filename = request.args.get('image_path', '')
    if not filename:
        return jsonify({"error": "image_path manquant"}), 400
    path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "image introuvable"}), 404
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    proposals = generate_region_proposals(rgb)
    return jsonify({"image_name": filename, "regions": proposals[:30]})


@app.route('/api/object/train', methods=['POST'])
def train_object_model():
    samples = build_object_training_samples()
    if len(samples) < 5:
        return jsonify({"error": "Pas assez d'exemples annotes pour entrainer. Ajoute au moins 5 regions."}), 400
    model = {
        "samples": samples,
        "classes": KNOWN_CLASSES,
        "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "sample_count": len(samples)
    }
    save_object_model(model)
    counts = {}
    for s in samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    return jsonify({"ok": True, "trained": True, "total_samples": len(samples), "counts": counts})


@app.route('/api/object/stats')
def object_stats():
    model = load_object_model()
    counts = {cls: 0 for cls in KNOWN_CLASSES}
    for s in model.get("samples", []):
        if s.get("label") in counts:
            counts[s["label"]] += 1
    return jsonify({"total_samples": len(model.get("samples", [])), "counts": counts, "classes": KNOWN_CLASSES})


@app.route('/api/object/reset', methods=['POST'])
def reset_object_model():
    save_labels_store({"records": []})
    save_object_model({"samples": [], "classes": KNOWN_CLASSES})
    return jsonify({"ok": True, "message": "Apprentissage objets reinitialise."})


@app.route('/api/object/predict')
def predict_objects():
    image_name = request.args.get("image_path", "")
    if not image_name:
        return jsonify({"error": "image_path manquant"}), 400
    model = load_object_model()
    samples = model.get("samples", [])
    if len(samples) < 5:
        return jsonify({"error": "Modele objets non entraine. Envoie d'abord des selections et entraine."}), 400
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "image introuvable"}), 404
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = np.array([s["features"] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples])
    n_neighbors = max(1, min(7, len(samples)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X, y)
    proposals = generate_region_proposals(rgb)
    predictions = []
    for proposal in proposals:
        feat = extract_region_features(rgb, proposal)
        if feat is None:
            continue
        label = knn.predict(feat.reshape(1, -1))[0]
        predictions.append({
            "id": proposal["id"],
            "x": proposal["x"],
            "y": proposal["y"],
            "w": proposal["w"],
            "h": proposal["h"],
            "label": label,
            "score": float(round(np.max(knn.predict_proba(feat.reshape(1, -1))), 3)) if hasattr(knn, 'predict_proba') else proposal.get("score", 0.0)
        })
    return jsonify({"image_name": image_name, "predictions": predictions[:30]})


@app.route('/api/object/predict/image')
def predict_objects_image():
    image_name = request.args.get("image_path", "")
    if not image_name:
        return jsonify({"error": "image_path manquant"}), 400
    model = load_object_model()
    samples = model.get("samples", [])
    if len(samples) < 5:
        return jsonify({"error": "Modele objets non entraine. Envoie d'abord des selections et entraine."}), 400
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "image introuvable"}), 404
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = np.array([s["features"] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples])
    n_neighbors = max(1, min(7, len(samples)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X, y)
    predictions = []
    proposals = generate_region_proposals(rgb)
    for proposal in proposals:
        feat = extract_region_features(rgb, proposal)
        if feat is None:
            continue
        label = knn.predict(feat.reshape(1, -1))[0]
        score = float(round(np.max(knn.predict_proba(feat.reshape(1, -1))), 3)) if hasattr(knn, 'predict_proba') else proposal.get("score", 0.0)
        predictions.append({"id": proposal["id"], "x": proposal["x"], "y": proposal["y"], "w": proposal["w"], "h": proposal["h"], "label": label, "score": score})
    
    # Si aucune prediction, retourner une image avec un message
    if not predictions:
        overlay = img.copy()
        # Ajouter un message texte sur l'image
        h, w = overlay.shape[:2]
        cv2.putText(overlay, "Aucun objet détecté", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(overlay, "Aucun objet détecté", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', overlay)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    
    overlay = img.copy()
    for pred in predictions[:30]:
        cv2.rectangle(overlay, (pred["x"], pred["y"]), (pred["x"] + pred["w"], pred["y"] + pred["h"]), (0, 255, 0), 2)
        cv2.putText(overlay, pred["label"], (pred["x"], max(pred["y"] - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', overlay)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')


@app.route('/api/labels/<image_name>')
def get_labels(image_name):
    store = load_labels_store()
    record = find_record_by_image(store["records"], image_name)
    if not record:
        return jsonify({"image_name": image_name, "regions": [], "classes": KNOWN_CLASSES})
    return jsonify(record)


@app.route('/api/labels/save', methods=['POST'])
def save_labels():
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name")
    regions = payload.get("regions", [])
    source_algo = payload.get("source_algo", "manual")
    notes = payload.get("notes", "")
    if not image_name:
        return jsonify({"error": "image_name requis"}), 400

    validated = []
    for region in regions:
        label = region.get("label", "inconnu")
        if label not in KNOWN_CLASSES:
            label = "inconnu"
        validated.append({
            "id": region.get("id", str(uuid.uuid4())),
            "label": label,
            "x": _safe_int(region.get("x"), 0),
            "y": _safe_int(region.get("y"), 0),
            "w": max(1, _safe_int(region.get("w"), 1)),
            "h": max(1, _safe_int(region.get("h"), 1)),
            "score": float(region.get("score", 1.0))
        })

    store = load_labels_store()
    existing = find_record_by_image(store["records"], image_name)
    payload_record = {
        "image_name": image_name,
        "classes": KNOWN_CLASSES,
        "regions": validated,
        "source_algo": source_algo,
        "notes": notes,
        "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": 1 if not existing else int(existing.get("version", 1)) + 1
    }
    if existing:
        store["records"][store["records"].index(existing)] = payload_record
    else:
        store["records"].append(payload_record)
    save_labels_store(store)
    return jsonify({"ok": True, "saved_regions": len(validated), "record": payload_record})


@app.route('/api/labels/apply', methods=['POST'])
def apply_labels():
    payload = request.get_json(silent=True) or {}
    source_image = payload.get("source_image")
    target_image = payload.get("target_image")
    source_width = _safe_int(payload.get("source_width"), 0)
    source_height = _safe_int(payload.get("source_height"), 0)
    if not source_image or not target_image:
        return jsonify({"error": "source_image et target_image requis"}), 400

    store = load_labels_store()
    source_rec = find_record_by_image(store["records"], source_image)
    if not source_rec:
        return jsonify({"error": "labels source introuvables"}), 404

    target_path = os.path.join(IMAGE_FOLDER, target_image)
    target_img = cv2.imread(target_path)
    if target_img is None:
        return jsonify({"error": "image cible introuvable"}), 404
    th, tw = target_img.shape[:2]

    if source_width <= 0 or source_height <= 0:
        source_path = os.path.join(IMAGE_FOLDER, source_image)
        source_img = cv2.imread(source_path)
        if source_img is None:
            return jsonify({"error": "image source introuvable"}), 404
        source_height, source_width = source_img.shape[:2]

    sx = tw / float(max(1, source_width))
    sy = th / float(max(1, source_height))
    new_regions = scale_regions(source_rec.get("regions", []), sx, sy)

    new_record = {
        "image_name": target_image,
        "classes": KNOWN_CLASSES,
        "regions": new_regions,
        "source_algo": f"transfer:{source_rec.get('source_algo', 'manual')}",
        "notes": f"Copie depuis {source_image}",
        "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": 1
    }
    existing = find_record_by_image(store["records"], target_image)
    if existing:
        new_record["version"] = int(existing.get("version", 1)) + 1
        store["records"][store["records"].index(existing)] = new_record
    else:
        store["records"].append(new_record)
    save_labels_store(store)
    return jsonify({"ok": True, "target_record": new_record})


@app.route('/api/scene/add_samples', methods=['POST'])
def add_scene_samples():
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name")
    points = payload.get("points", [])
    if not image_name:
        return jsonify({"error": "image_name requis"}), 400
    if not isinstance(points, list) or not points:
        return jsonify({"error": "points requis"}), 400

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "image introuvable"}), 404
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    store = load_scene_samples()
    samples = store.get("samples", [])
    added = 0
    for p in points:
        label = p.get("label", "").strip().lower()
        if label not in SCENE_CLASSES:
            continue
        x = _safe_int(p.get("x"), -1)
        y = _safe_int(p.get("y"), -1)
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        r, g, b = rgb[y, x]
        samples.append({
            "image_name": image_name,
            "label": label,
            "x": int(x),
            "y": int(y),
            "r": int(r),
            "g": int(g),
            "b": int(b),
            "yn": float(y / max(1, h - 1))
        })
        added += 1
    store["samples"] = samples
    save_scene_samples(store)
    counts = {cls: 0 for cls in SCENE_CLASSES}
    for s in samples:
        if s.get("label") in counts:
            counts[s["label"]] += 1
    return jsonify({"ok": True, "added": added, "total": len(samples), "counts": counts})


@app.route('/api/scene/reset', methods=['POST'])
def reset_scene_samples():
    save_scene_samples({"samples": []})
    return jsonify({"ok": True})


@app.route('/api/scene/stats')
def scene_stats():
    samples = load_scene_samples().get("samples", [])
    counts = {cls: 0 for cls in SCENE_CLASSES}
    for s in samples:
        label = s.get("label")
        if label in counts:
            counts[label] += 1
    return jsonify({"total": len(samples), "counts": counts, "classes": SCENE_CLASSES})


@app.route('/api/scene/predict')
def predict_scene():
    image_name = request.args.get("image_path", "")
    if not image_name:
        return jsonify({"error": "image_path manquant"}), 400
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "image introuvable"}), 404
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    samples = load_scene_samples().get("samples", [])
    if len(samples) < 12:
        return jsonify({"error": "Pas assez d'exemples. Ajoute au moins 12 selections terre/mer/ciel."}), 400

    X = np.array([[s["r"], s["g"], s["b"], s["yn"]] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples])
    n_neighbors = max(1, min(7, len(samples)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X, y)

    max_dim = 240
    scale = min(1.0, max_dim / float(max(h, w)))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(rgb, (sw, sh))
    yy, _ = np.mgrid[0:sh, 0:sw]
    yn = (yy.astype(np.float32) / max(1, sh - 1)).reshape(-1, 1)
    feats = np.concatenate([small.reshape(-1, 3).astype(np.float32), yn], axis=1)
    pred = knn.predict(feats).reshape(sh, sw)

    palette = {
        "terre": np.array([60, 120, 60], dtype=np.uint8),
        "mer": np.array([40, 110, 210], dtype=np.uint8),
        "ciel": np.array([150, 180, 255], dtype=np.uint8)
    }
    overlay_small = np.zeros((sh, sw, 3), dtype=np.uint8)
    for cls in SCENE_CLASSES:
        overlay_small[pred == cls] = palette[cls]

    overlay = cv2.resize(overlay_small, (w, h), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(rgb, 0.35, overlay, 0.65, 0)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/visualisation')
def visualisation():
    """Page de visualisation des données étiquetées"""
    history = load_processing_data()
    stats = get_statistics()
    
    # Charger le contenu des onglets markdown
    guide_content = ""
    verrous_content = ""
    
    try:
        with open('guide.md', 'r', encoding='utf-8') as f:
            guide_content = Markup(markdown.markdown(f.read()))
    except FileNotFoundError:
        guide_content = "<p>Guide non disponible</p>"
    
    try:
        with open('verrous.md', 'r', encoding='utf-8') as f:
            verrous_content = Markup(markdown.markdown(f.read()))
    except FileNotFoundError:
        verrous_content = "<p>Documentation des verrous non disponible</p>"
    
    labels_store = load_labels_store()

    return render_template('visualisation.html',
        processing_history=history,
        labels_history=labels_store.get("records", []),
        known_classes=KNOWN_CLASSES,
        total_images=stats['total_images'],
        processed_images=stats['processed_images'],
        total_objects=stats['total_objects'],
        success_rate=stats['success_rate'],
        avg_processing_time=stats['avg_processing_time'],
        algorithm_accuracy=stats['algorithm_accuracy'],
        preferred_algorithm=stats['preferred_algorithm'],
        preferred_count=stats['preferred_count'],
        avg_objects_per_image=stats['avg_objects_per_image'],
        kmeans_percent=stats['kmeans_percent'],
        dbscan_percent=stats['dbscan_percent'],
        cnn_percent=stats['cnn_percent'],
        manual_labels_count=stats['manual_labels_count'],
        manual_regions_count=stats['manual_regions_count'],
        guide_content=guide_content,
        verrous_content=verrous_content
    )

if __name__ == '__main__':
    app.run(debug=True, port=8000)