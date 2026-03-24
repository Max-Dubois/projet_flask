from tkinter import filedialog
from flask import Flask, render_template, request, send_from_directory, send_file
import os
import tkinter as tk
import cv2
import numpy as np
import io
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('accueil.html')

@app.route('/galerie', methods=['GET', 'POST'])
def galerie():
    images = []
    folder_path = ""
    
    if request.method == 'POST':
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory()
        root.destroy()

        if folder_path:
            valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(valid_exts):
                    images.append(filename)
                
    return render_template('galerie.html', images=images, folder_path=folder_path)

@app.route('/image/<path:full_path>')
def serve_image(full_path):
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)
    image_path = request.args.get('image_path')
    k_clusters = int(request.args.get('k', 5))

    img = cv2.imread(image_path)
    if img is None: return "Erreur lors du chargement", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    max_dim = 100
    scale = max_dim / max(h, w)
    img_small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    s_yy, s_xx = np.mgrid[0:sh, 0:sw]
    data_small = np.column_stack((img_small.reshape(-1, 3), s_xx.ravel(), s_yy.ravel()))
    model = AgglomerativeClustering(n_clusters=k_clusters, linkage='ward')
    labels_small = model.fit_predict(data_small)

    colors_map = np.zeros((k_clusters, 3), dtype=np.uint8)
    for i in range(k_clusters):
        mask = (labels_small == i)
        if np.any(mask):
            colors_map[i] = np.mean(img_small.reshape(-1, 3)[mask], axis=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(data_small, labels_small)
    
    f_yy, f_xx = np.mgrid[0:h, 0:w]
    data_full = np.column_stack((
        img_rgb.reshape(-1, 3), 
        (f_xx.ravel() * scale), 
        (f_yy.ravel() * scale)
    ))
    
    labels_full = knn.predict(data_full)
    image_finale = colors_map[labels_full].reshape(h, w, 3)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/dbscan', methods=['POST'])
def dbscan():
    image_path = request.form.get('image_path')
    k = request.form.get('k', 5)
    return render_template('resultatDBScan.html', image_path=image_path, k=k)

@app.route('/dbscan_image')
def generate_dbscan_image():
    image_path = request.args.get('image_path')
    k_clusters = int(request.args.get('k', 5))
    
    eps = 50 / k_clusters
    min_samples = 2

    img = cv2.imread(image_path)
    if img is None: return "Erreur lors du chargement", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    max_dim = 100
    scale = max_dim / max(h, w)
    img_small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    s_yy, s_xx = np.mgrid[0:sh, 0:sw]
    data_small = np.column_stack((img_small.reshape(-1, 3), s_xx.ravel(), s_yy.ravel()))
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_small = dbscan.fit_predict(data_small)

    n_clusters = len(set(labels_small)) - (1 if -1 in labels_small else 0)
    colors_map = np.zeros((n_clusters + 1, 3), dtype=np.uint8)
    
    for i in range(n_clusters):
        mask = (labels_small == i)
        if np.any(mask):
            colors_map[i] = np.mean(img_small.reshape(-1, 3)[mask], axis=0)
    
    colors_map[-1] = [128, 128, 128]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(data_small, labels_small)
    
    f_yy, f_xx = np.mgrid[0:h, 0:w]
    data_full = np.column_stack((
        img_rgb.reshape(-1, 3), 
        (f_xx.ravel() * scale), 
        (f_yy.ravel() * scale)
    ))
    
    labels_full = knn.predict(data_full)
    
    image_finale = np.zeros((h, w, 3), dtype=np.uint8)
    for i, label in enumerate(labels_full):
        if label == -1:
            image_finale.flat[i*3:(i+1)*3] = colors_map[-1]
        else:
            image_finale.flat[i*3:(i+1)*3] = colors_map[label]
    
    image_finale = image_finale.reshape(h, w, 3)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)