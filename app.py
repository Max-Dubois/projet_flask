from flask import Flask, render_template, request, send_from_directory, send_file
import os
import cv2
import numpy as np
import io
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# 📁 Configuration des dossiers
IMAGE_FOLDER = os.path.join('static', 'images')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

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
    k_clusters = int(request.args.get('k', 5))
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
    image_segmentee = colors[labels_full if algo == 'kmeans' else labels_full + 1].reshape(h, w, 3)

    # --- 5. CORRECTION MORPHOMATHÉMATIQUE (VERROU 3) ---
    # Nettoyage des petites impuretés sur le masque final
    kernel = np.ones((5,5), np.uint8)
    # Ouverture pour enlever le "poivre" (petits points isolés)
    image_finale = cv2.morphologyEx(image_segmentee, cv2.MORPH_OPEN, kernel)
    # Fermeture pour lisser les formes des bateaux
    image_finale = cv2.morphologyEx(image_finale, cv2.MORPH_CLOSE, kernel)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)