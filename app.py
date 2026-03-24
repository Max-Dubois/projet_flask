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
    """Génère l'image segmentée à la volée"""
    filename = request.args.get('image_path')
    k_clusters = int(request.args.get('k', 5))
    path = os.path.join(IMAGE_FOLDER, filename)
    
    img = cv2.imread(path)
    if img is None: return "Erreur", 400
    
    # 1. Prétraitement : CLAHE (Correction luminosité pour Adam)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_alt = cv2.merge((cl,a,b))
    img_rgb = cv2.cvtColor(img_alt, cv2.COLOR_LAB2RGB)
    
    h, w, _ = img_rgb.shape

    if algo == 'kmeans':
        # Clustering Classique K-Means
        pixel_values = img_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_values, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        image_finale = res.reshape(img_rgb.shape)

    elif algo == 'dbscan':
        # Clustering DBSCAN (votre algo original optimisé)
        max_dim = 100
        scale = max_dim / max(h, w)
        img_small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
        sh, sw, _ = img_small.shape
        s_yy, s_xx = np.mgrid[0:sh, 0:sw]
        data_small = np.column_stack((img_small.reshape(-1, 3), s_xx.ravel(), s_yy.ravel()))
        
        db = DBSCAN(eps=(50/k_clusters), min_samples=3).fit(data_small)
        labels_small = db.labels_
        
        # Coloration par moyenne
        n_clusters = len(set(labels_small)) - (1 if -1 in labels_small else 0)
        colors = np.array([np.mean(img_small.reshape(-1,3)[labels_small==i], axis=0) if i!=-1 else [128,128,128] for i in range(-1, n_clusters)], dtype=np.uint8)
        
        # Upscaling via KNN
        knn = KNeighborsClassifier(n_neighbors=1).fit(data_small, labels_small)
        f_yy, f_xx = np.mgrid[0:h, 0:w]
        data_full = np.column_stack((img_rgb.reshape(-1, 3), f_xx.ravel()*scale, f_yy.ravel()*scale))
        labels_full = knn.predict(data_full)
        image_finale = colors[labels_full + 1].reshape(h, w, 3)

    else: # Placeholder pour Auto-encodeur (CNN)
        image_finale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # En attendant le modèle

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)