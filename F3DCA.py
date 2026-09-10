"""
Clustering Analysis Application with GUI
"""

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CPU_ONLY = os.environ.get('F3DCA_CPU_ONLY', '0') == '1'
if CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import random
import threading
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "savefig.dpi": 600
})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from sklearn.cluster import KMeans, Birch, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, f1_score, adjusted_rand_score, accuracy_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, f1_score

from tensorflow import keras
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Input, Dense, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import set_random_seed

import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from numba import jit
except Exception:
    jit = None
import tensorflow as tf
if CPU_ONLY:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass


def tensor_operation(data):
    return data * 2


class ClusteringApp:
    
    def __init__(self, master):
        self.master = master
        self.master.title("Clustering Analysis App")
        
        try:
            if os.path.exists('logo_image.ico'):
                self.master.iconbitmap('logo_image.ico')
        except Exception:
            pass
        self.file_names = []
        self.subset_autoencoder = tk.DoubleVar(value=0.1)
        self.analysis_running = False
        self.trajectory_running = False
        self._analysis_subset_percentage = 0.1
        
        
        width = self.master.winfo_screenwidth()
        height = self.master.winfo_screenheight()
        
        self.master.geometry("%dx%d" % (width, height))
        
        self.master.state("zoomed")
        self.master.configure(bg="#f0f0f0")
        self.master.option_add("*Font", "Helvetica 15")
        self.master.option_add("*Button.Background", "#4CAF50")
        self.master.option_add("*Button.Foreground", "#ffffff")
        
        
        background_path = 'background_img.jpg' if os.path.exists('background_img.jpg') else None
        if background_path:
            background_image = Image.open(background_path).resize((width, height), Image.LANCZOS)
        else:
            background_image = Image.new('RGB', (width, height), '#07111f')
        self.background_photo = ImageTk.PhotoImage(background_image)

        
        self.background_label = tk.Label(master, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)
       
        
        self.background_label = tk.Label(master, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        
        
        self.control_panel = tk.Frame(master, bg="#061426", bd=2, relief="ridge",
                                      highlightbackground="#00AEEF", highlightthickness=1)
        self.control_panel.place(relx=0.035, rely=0.13, relwidth=0.31, relheight=0.24)

        label_style = {"bg": "#061426", "fg": "#FFB000", "font": ("Helvetica", 12, "bold")}
        entry_style = {"font": ("Helvetica", 12), "bg": "white", "fg": "black", "relief": "sunken"}

        tk.Label(self.control_panel, text="Input files number:", **label_style).grid(
            row=0, column=0, sticky="w", pady=(10, 6), padx=10
        )
        self.entry_num_files = tk.Entry(self.control_panel, **entry_style)
        self.entry_num_files.grid(row=0, column=1, sticky="we", pady=(10, 6), padx=10)

        
        # subsampling control is intentionally not shown because it is not used by
        # the trajectory-window analysis.
        self.control_panel.grid_columnconfigure(1, weight=1)

        self.button_panel = tk.Frame(master, bg="#061426", bd=2, relief="ridge",
                                     highlightbackground="#00AEEF", highlightthickness=1)
        self.button_panel.place(relx=0.37, rely=0.13, relwidth=0.31, relheight=0.24)

        btn_style = {
            "font": ("Helvetica", 11, "bold"),
            "bg": "#1F9D42",
            "fg": "white",
            "activebackground": "#2EC95C",
            "activeforeground": "white",
            "relief": "raised",
            "bd": 3,
        }

        self.btn_generate_entries = tk.Button(
            self.button_panel, text="Generate Entries", command=self.generate_entries, **btn_style
        )
        self.btn_generate_entries.grid(row=0, column=0, sticky="we", pady=8, padx=8)

        self.btn_plot_folder = tk.Button(
            self.button_panel, text="Select Output Folder", command=self.browse_plot_folder, **btn_style
        )
        self.btn_plot_folder.grid(row=0, column=1, sticky="we", pady=8, padx=8)

        self.btn_results = tk.Button(
            self.button_panel, text="Run Final Trajectory Analysis", command=self.run_trajectory_analysis, **btn_style
        )
        self.btn_results.grid(row=1, column=0, columnspan=2, sticky="we", pady=8, padx=8)

        self.button_panel.grid_columnconfigure(0, weight=1)
        self.button_panel.grid_columnconfigure(1, weight=1)

        self.entry_file_names = []
        self.btn_browse_files = []
        
        self.generate_entries()
        
        
        script_dir = os.getcwd()
        os.chdir(script_dir)
        
        self.plot_folder = "plots"
        os.makedirs(self.plot_folder, exist_ok=True)
        
        self.plot_images = []
        
        
        self.seed = 55  
        random.seed(self.seed)
        np.random.seed(self.seed)
        set_random_seed(self.seed)  


    def generate_entries(self):
        num_files_str = self.entry_num_files.get()
        num_files = int(num_files_str) if num_files_str else 3

        if hasattr(self, "file_panel") and self.file_panel.winfo_exists():
            self.file_panel.destroy()

        self.file_panel = tk.Frame(self.master, bg="#061426", bd=2, relief="ridge",
                                   highlightbackground="#00AEEF", highlightthickness=1)
        self.file_panel.place(relx=0.035, rely=0.39, relwidth=0.31, relheight=min(0.08 + 0.055 * num_files, 0.45))

        self.entry_file_names = []
        self.btn_browse_files = []

        label_style = {"bg": "#061426", "fg": "#FFB000", "font": ("Helvetica", 10, "bold")}
        entry_style = {"font": ("Helvetica", 9), "bg": "white", "fg": "black", "relief": "sunken"}
        btn_style = {
            "font": ("Helvetica", 9, "bold"),
            "bg": "#1F9D42",
            "fg": "white",
            "activebackground": "#2EC95C",
            "activeforeground": "white",
            "relief": "raised",
            "bd": 2,
        }

        for i in range(num_files):
            tk.Label(self.file_panel, text=f"Browse filename {i + 1}:", **label_style).grid(
                row=i, column=0, sticky="w", pady=5, padx=6
            )
            entry_file_name = tk.Entry(self.file_panel, **entry_style)
            entry_file_name.grid(row=i, column=1, sticky="we", pady=5, padx=6)
            self.entry_file_names.append(entry_file_name)

            btn_browse = tk.Button(
                self.file_panel, text=f"Browse file {i + 1}", command=lambda i=i: self.browse_file(i), **btn_style
            )
            btn_browse.grid(row=i, column=2, sticky="we", pady=5, padx=6)
            self.btn_browse_files.append(btn_browse)

        self.file_panel.grid_columnconfigure(1, weight=1)

    def browse_file(self, index):
        selected = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not selected:
            return
        while len(self.file_names) <= index:
            self.file_names.append("")
        self.file_names[index] = selected
        self.entry_file_names[index].delete(0, tk.END)
        self.entry_file_names[index].insert(0, selected)
        messagebox.showinfo("File Selected", f"Selected File {index + 1}: {selected}")

    def browse_plot_folder(self):
        
        plot_folder = filedialog.askdirectory()
        if plot_folder:
            
            self.plot_folder = plot_folder
            messagebox.showinfo("Plot Output Folder", f"Selected Plot Output Folder: {self.plot_folder}")
    
    
    def display_elbow_plot(self):
        self.display_plot("Elbow Plot", f"{self.plot_folder}/elbow_plot.png", f"{self.plot_folder}/elbow_plot_EncodedData.png")

    def display_clusters_plot(self):
        self.display_plot("Clusters Plot", f"{self.plot_folder}/clustering_methods.png", f"{self.plot_folder}/clustering_methods_EncodedData.png")

    def display_cross_clusters_silhouette_plot(self):
        self.display_plot("Cross Clusters Silhouette Plot", f"{self.plot_folder}/cross_cluster_plots_Silhouette.png", f"{self.plot_folder}/cross_cluster_plots_Silhouette_EncodedData.png")

    def display_cross_clusters_davies_plot(self):
        self.display_plot("Cross Clusters Davies Bouldin Plot", f"{self.plot_folder}/cross_cluster_plots_Davies.png", f"{self.plot_folder}/cross_cluster_plots_Davies_EncodedData.png")
    
    def display_dendrogram_silhouette(self):
        self.display_plot("Dendrogram Silhouette", f"{self.plot_folder}/dendrogram_Silhouette_scores.png", f"{self.plot_folder}/dendrogram_Silhouette_scores_EncodedData.png")

    def display_dendrogram_davies(self):
        self.display_plot("Dendrogram Davies Bouldin", f"{self.plot_folder}/dendrogram_Davies_scores.png", f"{self.plot_folder}/dendrogram_Davies_scores_EncodedData.png")
    
    
    def display_plot(self, button_name, before_encoding_filename, after_encoding_filename):
        
        for widget in self.result_window.winfo_children():
            if isinstance(widget, tk.Label) and widget.grid_info().get("row") == 7 and widget.grid_info().get("column") == 1:
                widget.destroy()
                
        
        title_label = tk.Label(self.result_window, text=button_name, font=('Helvetica', 14, 'bold'))
        title_label.grid(row=7, column=1, padx=10)
    
        
        before_encoding_image = Image.open(before_encoding_filename)
        after_encoding_image = Image.open(after_encoding_filename)
    
        
        before_encoding_image = before_encoding_image.resize((370, 370), Image.LANCZOS)
        after_encoding_image = after_encoding_image.resize((370, 370), Image.LANCZOS)
    
        
        before_encoding_tk_image = ImageTk.PhotoImage(before_encoding_image)
        after_encoding_tk_image = ImageTk.PhotoImage(after_encoding_image)
    
        
        before_encoding_label = tk.Label(self.result_window, image=before_encoding_tk_image)
        before_encoding_label.image = before_encoding_tk_image  
        before_encoding_label.grid(row=10, column=0, padx=10)
    
        after_encoding_label = tk.Label(self.result_window, image=after_encoding_tk_image)
        after_encoding_label.image = after_encoding_tk_image  
        after_encoding_label.grid(row=10, column=2, padx=10)
        
    def save_clustering_results(self, silhouette_scores, davies_scores, data_type="before"):
        methods = ["KMeans", "Birch", "GMM", "Spectral"]
        silhouette_scores = np.asarray(silhouette_scores)
        davies_scores = np.asarray(davies_scores)
        if silhouette_scores.ndim == 1:
            silhouette_scores = silhouette_scores.reshape(1, -1)
        if davies_scores.ndim == 1:
            davies_scores = davies_scores.reshape(1, -1)

        if silhouette_scores.shape[0] == 1:
            df_class = pd.DataFrame([{
                'Dataset': 'Combined',
                'KMeans_Silhouette': silhouette_scores[0, 0],
                'Birch_Silhouette': silhouette_scores[0, 1],
                'GMM_Silhouette': silhouette_scores[0, 2],
                'Spectral_Silhouette': silhouette_scores[0, 3],
                'KMeans_Davies': davies_scores[0, 0],
                'Birch_Davies': davies_scores[0, 1],
                'GMM_Davies': davies_scores[0, 2],
                'Spectral_Davies': davies_scores[0, 3],
            }])
        else:
            file_names = [os.path.splitext(os.path.basename(file))[0] for file in self.file_names]
            class_scores = []
            for i in range(min(len(file_names), silhouette_scores.shape[0])):
                class_scores.append({
                    'Dataset': file_names[i],
                    'KMeans_Silhouette': silhouette_scores[i, 0],
                    'Birch_Silhouette': silhouette_scores[i, 1],
                    'GMM_Silhouette': silhouette_scores[i, 2],
                    'Spectral_Silhouette': silhouette_scores[i, 3],
                    'KMeans_Davies': davies_scores[i, 0],
                    'Birch_Davies': davies_scores[i, 1],
                    'GMM_Davies': davies_scores[i, 2],
                    'Spectral_Davies': davies_scores[i, 3],
                })
            df_class = pd.DataFrame(class_scores)

        df_method = pd.DataFrame({
            'Method': methods,
            'Avg_Silhouette': np.mean(silhouette_scores, axis=0),
            'Avg_Davies': np.mean(davies_scores, axis=0),
        })

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df_class.to_csv(f"{self.plot_folder}/class_scores_{data_type}_{timestamp}.csv", index=False)
        df_method.to_csv(f"{self.plot_folder}/method_scores_{data_type}_{timestamp}.csv", index=False)

    def subset_data_consistently(self, data, subset_percentage):
        """Select consistent data subset"""
        np.random.seed(55)  
        subset_size = int(subset_percentage * len(data))
        indices = np.random.choice(len(data), subset_size, replace=False)
        return data.iloc[indices]
    
    
    def show_results(self):
        if self.analysis_running:
            messagebox.showinfo("Analysis", "Analysis is already running.")
            return
        if not self.file_names or not all(self.file_names):
            messagebox.showwarning("Warning", "Please select input files.")
            return
        
        
        
        try:
            subset_percentage = float(self.subset_autoencoder.get())
            if not 0 < subset_percentage <= 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid value", "Analysis subsampling fraction must be greater than 0 and less than or equal to 1.")
            return

        self._analysis_subset_percentage = subset_percentage
        self._analysis_file_names = list(self.file_names)
        self._analysis_plot_folder = str(self.plot_folder)
        self.analysis_running = True
        self.btn_results.config(state="disabled", text="Analysis Running...")
        self.master.update_idletasks()
        threading.Thread(target=self._analysis_worker_entry, daemon=True).start()

    def _analysis_worker_entry(self):
        try:
            self._show_results_worker()
        except Exception as exc:
            details = traceback.format_exc()
            print(details)
            self.master.after(0, lambda e=str(exc): self._analysis_failed(e))

    def _analysis_failed(self, error_message):
        self.analysis_running = False
        if self.master.winfo_exists():
            self.btn_results.config(state="normal", text="Show Results")
        messagebox.showerror("Analysis Error", error_message)

    def _analysis_finished(self):
        self.analysis_running = False
        if self.master.winfo_exists():
            self.btn_results.config(state="normal", text="Show Results")
        messagebox.showinfo("Results", "Clustering results can be displayed.")
        self._open_results_window()

    def _show_results_worker(self):
        
        print("[F3DCA] Analysis started.", flush=True)
        dfs = []
        file_names = list(self._analysis_file_names)

        def canonical_group_name(file_name):
            name = os.path.splitext(os.path.basename(file_name))[0]
            low = name.lower().replace('_', ' ').strip()

            
            if 'male & male' in low or 'male and male' in low:
                return 'Male-Male'
            if 'female & female' in low or 'female and female' in low:
                return 'Female-Female'
            if 'male & female' in low or 'male and female' in low:
                return 'Male-Female'

            
            for token, label in [('00-01', '0-1 min'), ('05-06', '5-6 min'),
                                 ('10-11', '10-11 min'), ('15-16', '15-16 min'),
                                 ('20-21', '20-21 min')]:
                if token in low:
                    return label

            
            if 'control' in low:
                return 'Control'
            if '0.25% etoh' in low or '0.25% ethanol' in low:
                return '0.25% Ethanol'
            if ('1% etoh' in low or '1.0% etoh' in low or
                    '1% ethanol' in low or '1.0% ethanol' in low):
                return '1.0% Ethanol'

            
            if 'cat fish' in low or 'catfish' in low:
                return 'Glass Catfish'
            if 'ory combined' in low or 'medaka' in low or 'oryzias' in low:
                return 'Medaka'
            if 'zebrafish' in low:
                return 'Zebrafish'

            
            if 'unamputated' in low:
                return 'Unamputated'
            if 'caudal fin amputated' in low and ('lidocaine' in low or '5 ppm' in low):
                return 'Caudal Fin Amputated + Lidocaine'
            if low in {'caudal fin', 'caudal fin amputated', 'caudal fin amputation'} or 'caudal fin.xlsx' in low:
                return 'Caudal Fin Amputated'
            if 'lidocaine' in low or '5 ppm lidocaine' in low:
                return 'Lidocaine only'

            
            import re
            clean = re.sub(r'\s*\((?:1st|2nd|3rd|[0-9]+(?:st|nd|rd|th)?)\s+batch\)\s*$', '', name, flags=re.I)
            clean = re.sub(r'\s+(?:0?[1-9]|[1-9][0-9])\s*$', '', clean).strip(' -_')
            return clean

        
        
        print(f"[F3DCA] Loading {len(file_names)} input file(s)...", flush=True)
        for file_name in file_names:
            label = canonical_group_name(file_name)
            xls = pd.ExcelFile(file_name)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                df = df.iloc[:, :3]
                df.columns = ['X', 'Y', 'Z']
                df['label'] = label  
                
                dfs.append(df)
        
        concatenated_data = pd.concat(dfs, ignore_index=True)
        
        
        csv_filename = os.path.join(self.plot_folder, 'concatenated_data.csv')
        concatenated_data.to_csv(csv_filename, index=False)
        
        
        
        
        
        
        def DataPreprocessing(subset_data):
          z_scores = np.abs(zscore(subset_data))
          
          threshold = 3  
          
          rows_to_keep = (z_scores < threshold).all(axis=1)
          
          subset_data = subset_data[rows_to_keep]
        
          
          scaler = StandardScaler()
        
          
          normalized_features = scaler.fit_transform(subset_data)
        
          
          data_for_clustering = pd.DataFrame(normalized_features, columns=subset_data.columns)
        
          return data_for_clustering, rows_to_keep
        
        
        
        
        
        subset_percentage = float(self._analysis_subset_percentage)
        if not 0 < subset_percentage <= 1:
            raise ValueError("Analysis subsampling fraction must be greater than 0 and less than or equal to 1.")
        print(f"[F3DCA] Preprocessing with retained fraction={subset_percentage:.3f}...", flush=True)
        subset_data = self.subset_data_consistently(concatenated_data, subset_percentage)
        subset_data = subset_data.dropna()
        data_for_clustering = subset_data[['X', 'Y', 'Z']]
        data_for_clustering, rows_to_keep = DataPreprocessing(data_for_clustering)
        subset_data = subset_data.loc[rows_to_keep].copy()
        labels = subset_data.label.astype(str)
        preferred_group_order = [
            'Zebrafish', 'Medaka', 'Glass Catfish',
            'Male-Male', 'Female-Female', 'Male-Female',
            'Unamputated', 'Caudal Fin Amputated',
            'Caudal Fin Amputated + Lidocaine', 'Lidocaine only',
            '0-1 min', '5-6 min', '10-11 min', '15-16 min', '20-21 min',
            'Control', '0.25% Ethanol', '1.0% Ethanol'
        ]
        present_groups = list(pd.unique(labels))
        group_names = [g for g in preferred_group_order if g in present_groups]
        group_names.extend([g for g in present_groups if g not in group_names])
        group_to_id = {name: i for i, name in enumerate(group_names)}
        Labels = labels.map(group_to_id).to_numpy(dtype=int)
        analysis_subset = subset_data.copy()
        analysis_raw = data_for_clustering.copy()
        analysis_labels = Labels.copy()

        
        
        
        
        evaluation_cap = 3000
        evaluation_size = min(evaluation_cap, len(analysis_raw))
        evaluation_rng = np.random.default_rng(55)
        if len(analysis_raw) > evaluation_size:
            evaluation_indices = np.sort(
                evaluation_rng.choice(len(analysis_raw), size=evaluation_size, replace=False)
            )
        else:
            evaluation_indices = np.arange(len(analysis_raw))

        raw_eval = analysis_raw.iloc[evaluation_indices].reset_index(drop=True)
        labels_eval = analysis_labels[evaluation_indices]
        print(
            f"[F3DCA] Retained {len(analysis_raw):,} observations; "
            f"using {evaluation_size:,} fixed observations for all clustering methods/figures.",
            flush=True
        )

        np.shape(Labels)
        
        
        
        
        def Generate_Elbow(data, encoded_bool):
          num_clusters_range = range(1, 11)
        
          wcss = []
        
          
          for num_clusters in num_clusters_range:
              kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=55)
              kmeans.fit(data)
              wcss.append(kmeans.inertia_)  
        
          
          plt.figure(figsize=(10, 6))
          plt.plot(num_clusters_range, wcss, marker='o')
          plt.title('Elbow Method')
          plt.xlabel('Number of Clusters')
          plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
          
          
          if encoded_bool== False:
              
              plt.savefig(f"{self.plot_folder}/elbow_plot.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
              plt.close()
    
          else:
              
              plt.savefig(f"{self.plot_folder}/elbow_plot_EncodedData.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
              plt.close()
    
        print("[F3DCA] Preprocessing complete. Running raw-data elbow/clustering...", flush=True)
        Generate_Elbow(raw_eval, False)
        
        
        
        def Generate_Scores(data, labels):
          silhouette_avg = silhouette_score(data, labels)
          davies_bouldin = davies_bouldin_score(data, labels)
        
          return silhouette_avg, davies_bouldin
        
        def posthoc_cluster_name_map(predicted_labels, true_labels):
            cluster_ids = np.unique(predicted_labels)
            true_ids = np.unique(true_labels)
            contingency = np.zeros((len(cluster_ids), len(true_ids)), dtype=int)
            for i, cluster_id in enumerate(cluster_ids):
                for j, true_id in enumerate(true_ids):
                    contingency[i, j] = np.sum((predicted_labels == cluster_id) & (true_labels == true_id))
            row_ind, col_ind = linear_sum_assignment(-contingency)
            true_name_lookup = {i: str(name) for i, name in enumerate(group_names)}
            mapping = {}
            for r, c in zip(row_ind, col_ind):
                mapping[cluster_ids[r]] = true_name_lookup.get(true_ids[c], f"Group {true_ids[c]}")
            for cluster_id in cluster_ids:
                mapping.setdefault(cluster_id, f"Cluster {cluster_id}")
            return mapping

        def ClusteringMethods(data_for_clustering, n_clusters, true_labels, encoded_bool=False):
            np.random.seed(55)

            print(f"[F3DCA] {'Encoded' if encoded_bool else 'Raw'}: K-Means...", flush=True)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=55)
            kmeans_labels = kmeans.fit_predict(data_for_clustering)
            silhouette_k, davies_bouldin_k = Generate_Scores(data_for_clustering, kmeans_labels)

            print(f"[F3DCA] {'Encoded' if encoded_bool else 'Raw'}: BIRCH...", flush=True)
            birch = Birch(n_clusters=n_clusters, threshold=0.01)
            birch_labels = birch.fit_predict(data_for_clustering)
            silhouette_b, davies_bouldin_b = Generate_Scores(data_for_clustering, birch_labels)

            print(f"[F3DCA] {'Encoded' if encoded_bool else 'Raw'}: GMM...", flush=True)
            gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=55)
            gmm_labels = gmm.fit_predict(data_for_clustering)
            silhouette_gmm, davies_bouldin_gmm = Generate_Scores(data_for_clustering, gmm_labels)

            print(f"[F3DCA] {'Encoded' if encoded_bool else 'Raw'}: Spectral (capped evaluation subset)...", flush=True)
            spectral = SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=55, affinity='rbf')
            spectral_labels = spectral.fit_predict(data_for_clustering)
            silhouette_spec, davies_bouldin_spec = Generate_Scores(data_for_clustering, spectral_labels)

            data_array = np.asarray(data_for_clustering)
            fig = plt.figure(figsize=(15.5, 15.5))
            plot_specs = [
                ('KMeans', kmeans_labels),
                ('Birch', birch_labels),
                ('GMM', gmm_labels),
                ('Spectral', spectral_labels),
            ]
            axis_suffix = ' Encoded' if encoded_bool else ''
            mapping_rows = []
            for plot_index, (title, predicted) in enumerate(plot_specs, start=1):
                ax = fig.add_subplot(2, 2, plot_index, projection='3d')
                name_map = posthoc_cluster_name_map(predicted, true_labels)
                for cluster_id, mapped_name in name_map.items():
                    mapping_rows.append({
                        'Method': title,
                        'Cluster_ID': int(cluster_id),
                        'Posthoc_File_Name': mapped_name
                    })
                
                
                
                canonical_names = list(group_names)
                name_to_cluster = {mapped_name: cluster_id for cluster_id, mapped_name in name_map.items()}

                for color_index, mapped_name in enumerate(canonical_names):
                    if mapped_name not in name_to_cluster:
                        continue
                    cluster_id = name_to_cluster[mapped_name]
                    mask = predicted == cluster_id
                    ax.scatter(
                        data_array[mask, 0], data_array[mask, 1], data_array[mask, 2],
                        s=44, alpha=0.92, edgecolors="black", linewidths=0.35, depthshade=True,
                        color=f"C{color_index}",
                        label=mapped_name
                    )

                
                
                mapped_cluster_ids = set(name_to_cluster.values())
                for cluster_id in np.unique(predicted):
                    if cluster_id in mapped_cluster_ids:
                        continue
                    mask = predicted == cluster_id
                    ax.scatter(
                        data_array[mask, 0], data_array[mask, 1], data_array[mask, 2],
                        s=44, alpha=0.92, edgecolors="black", linewidths=0.35, depthshade=True,
                        label=name_map[cluster_id]
                    )

                ax.set_title(title, fontsize=22, fontweight='bold')
                ax.set_xlabel('X' + axis_suffix)
                ax.set_ylabel('Y' + axis_suffix)
                ax.set_zlabel('Z' + axis_suffix)
                ax.legend()

            plt.tight_layout()
            plot_name = 'clustering_methods_EncodedData.png' if encoded_bool else 'clustering_methods.png'
            plt.savefig(f"{self.plot_folder}/{plot_name}", dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close()
            mapping_name = 'cluster_name_mapping_after.csv' if encoded_bool else 'cluster_name_mapping_before.csv'
            pd.DataFrame(mapping_rows).to_csv(os.path.join(self.plot_folder, mapping_name), index=False)

            score_row = np.array([[
                silhouette_k, silhouette_b, silhouette_gmm, silhouette_spec
            ]])
            davies_row = np.array([[
                davies_bouldin_k, davies_bouldin_b, davies_bouldin_gmm, davies_bouldin_spec
            ]])
            return score_row, davies_row

        n_clusters = int(len(group_names))
        print(f"[F3DCA] Biological groups ({n_clusters}): {', '.join(group_names)}", flush=True)
        silhouette_scores, davies_scores = ClusteringMethods(
            raw_eval, n_clusters, labels_eval, encoded_bool=False
        )

        def CrossClusterPlots(scores, type, encoded_bool):
            methods = ['kMeans', 'Birch', 'Gmm', 'Spectral']
            colors = ['blue', 'orange', 'green', 'red']
        
            
            plt.figure(figsize=(10, 6))
            plt.bar(methods, scores, color=colors)
            plt.title(type + ' Scores for Different Clustering Methods')
            plt.xlabel('Clustering Methods')
            plt.ylabel(type + ' Score')
            plt.ylim(0, np.max(scores) + 0.1)  
            
            
            if encoded_bool== False:
                
                plt.savefig(f"{self.plot_folder}/cross_cluster_plots_{type}.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
                plt.close()
          
            else:
                
                plt.savefig(f"{self.plot_folder}/cross_cluster_plots_{type}_EncodedData.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
                plt.close()
          
        
        silhouette_scores = np.array(silhouette_scores)
        davies_scores = np.array(davies_scores)

        
        CrossClusterPlots(np.mean(silhouette_scores, axis = 0), 'Silhouette', False)
        CrossClusterPlots(np.mean(davies_scores, axis = 0), 'Davies', False)
        
        
        linkage_matrix = linkage(davies_scores.T, method='ward')  
        
        
        dendrogram(linkage_matrix, labels=["KMeans", "Birch", "GMM", "Spectral"])
        
        plt.title('Dendrogram based on Davies Scores')
        plt.xlabel('Clustering Algorithms')
        plt.ylabel('Distance')
        
        
        
        plt.savefig(f"{self.plot_folder}/dendrogram_Davies_scores.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close()
        
        
        linkage_matrix = linkage(silhouette_scores.T, method='ward')  
        dendrogram(linkage_matrix, labels=["KMeans", "Birch", "GMM", "Spectral"])
        
        plt.title('Dendrogram based on Silhouette Scores')
        plt.xlabel('Clustering Algorithms')
        plt.ylabel('Distance')
        
        
        
        plt.savefig(f"{self.plot_folder}/dendrogram_Silhouette_scores.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close()
        
        
        self.save_clustering_results(silhouette_scores, davies_scores, "before")
        
        
        subset_data = analysis_subset.copy()
        data_for_encoding = np.asarray(analysis_raw)
        Labels = analysis_labels.copy()
        
        data_for_encoding
        
        
        data_for_encoding_df = pd.DataFrame(data_for_encoding, columns=['X', 'Y', 'Z'])

        
        csv_file_name = os.path.join(self.plot_folder, 'data_for_encoding.csv')
        data_for_encoding_df.to_csv(csv_file_name, index=False)
        
        
        
        
        
        AE_SEED = 55
        tf.keras.utils.set_random_seed(AE_SEED)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        def build_conv1d_autoencoder(data_size, code_size=3):
            encoder = keras.Sequential([
                keras.layers.Input(shape=data_size),
                keras.layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Flatten(),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(code_size, activation="linear")
            ], name="encoder")

            decoder = keras.Sequential([
                keras.layers.Input(shape=(code_size,)),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(data_size[0] * 32, activation="relu"),
                keras.layers.Reshape((data_size[0], 32)),
                keras.layers.Conv1DTranspose(32, kernel_size=3, activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1DTranspose(16, kernel_size=3, activation="relu", padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(1, kernel_size=3, activation="linear", padding="same")
            ], name="decoder")

            return encoder, decoder

        data_for_encoding = np.asarray(data_for_encoding, dtype=np.float32)
        data_for_encoding = data_for_encoding.reshape((-1, 3, 1))

        data_size = (3, 1)
        code_size = 3

        encoder, decoder = build_conv1d_autoencoder(data_size, code_size)

        inp = keras.layers.Input(shape=data_size, name="xyz_input")
        code = encoder(inp)
        reconstruction = decoder(code)
        autoencoder = keras.models.Model(
            inputs=inp,
            outputs=reconstruction,
            name="F3DCA_autoencoder"
        )

        optimizer = Adam(learning_rate=1e-4)
        autoencoder.compile(loss="mse", optimizer=optimizer)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            min_delta=1e-6,
            restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        print(
            f"[F3DCA] Starting autoencoder "
            f"(XYZ input -> {code_size}D latent, seed={AE_SEED})...",
            flush=True
        )

        history = autoencoder.fit(
            x=data_for_encoding,
            y=data_for_encoding,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(
            os.path.join(self.plot_folder, "autoencoder_training_history.csv"),
            index=False
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(history_df["loss"], label="Training loss")
        if "val_loss" in history_df.columns:
            ax.plot(history_df["val_loss"], label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Autoencoder Training History")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.plot_folder, "autoencoder_training_history.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.35
        )
        plt.close(fig)

        with open(
            os.path.join(self.plot_folder, "autoencoder_configuration.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write("Random seed: 55\n")
            f.write("Input: standardized XYZ coordinate observation, shape 3 x 1\n")
            f.write("Encoder Conv1D filters: 16, 32\n")
            f.write("Latent dimensions: 3\n")
            f.write("Latent activation: linear\n")
            f.write("Loss: mean squared error\n")
            f.write("Optimizer: Adam, initial learning rate 1e-4\n")
            f.write("Batch size: 64\n")
            f.write("Maximum epochs: 150\n")
            f.write("Validation split: 0.20\n")
            f.write("Early-stopping patience: 12\n")
            f.write("Evaluation subset maximum: 3000 observations, seed 55\n")

        print("[F3DCA] Autoencoder training complete. Encoding data...", flush=True)
        encoded_data = encoder.predict(
            data_for_encoding,
            batch_size=256,
            verbose=0
        )

        latent_scaler = StandardScaler()
        encoded_data_normalized = latent_scaler.fit_transform(encoded_data)

        encoded_columns = [f"Latent_{i+1}" for i in range(code_size)]
        encoded_data_df = pd.DataFrame(
            encoded_data_normalized,
            columns=encoded_columns
        )
        csv_file_name = os.path.join(self.plot_folder, "encoded_data.csv")
        encoded_data_df.to_csv(csv_file_name, index=False)

        
        encoded_eval = encoded_data_normalized[evaluation_indices]

        print(
            f"[F3DCA] Encoded data saved. Running all four clustering methods on "
            f"the same {len(encoded_eval):,}-point evaluation subset...",
            flush=True
        )
        Generate_Elbow(encoded_eval, True)
        
        n_clusters = int(len(group_names))
        silhouette_scores, davies_scores = ClusteringMethods(
            encoded_eval, n_clusters, labels_eval, encoded_bool=True
        )
            
        
        CrossClusterPlots(np.mean(silhouette_scores, axis = 0), 'Silhouette', True)
        CrossClusterPlots(np.mean(davies_scores, axis = 0), 'Davies', True)
        
        
        linkage_matrix = linkage(davies_scores.T, method='ward')  
        
        
        dendrogram(linkage_matrix, labels=["KMeans", "Birch", "GMM", "Spectral"])
        
        plt.title('Dendrogram based on Davies Scores')
        plt.xlabel('Clustering Algorithms')
        plt.ylabel('Distance')
        
        
        
        plt.savefig(f"{self.plot_folder}/dendrogram_Davies_scores_EncodedData.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close()
  
        
        linkage_matrix = linkage(silhouette_scores.T, method='ward')  
        dendrogram(linkage_matrix, labels=["KMeans", "Birch", "GMM", "Spectral"])
        
        plt.title('Dendrogram based on Silhouette Scores')
        plt.xlabel('Clustering Algorithms')
        plt.ylabel('Distance')
        
        
        
        plt.savefig(f"{self.plot_folder}/dendrogram_Silhouette_scores_EncodedData.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close()
        
        
        self.save_clustering_results(silhouette_scores, davies_scores, "after")
        
        
        print("[F3DCA] Analysis complete. Results saved.", flush=True)
        self.master.after(0, self._analysis_finished)

    def _open_results_window(self):
        
        self.master.withdraw()
        self.result_window = tk.Toplevel(self.master)
        self.result_window.title("Clustering Results")
        
        try:
            if os.path.exists('logo_image.ico'):
                self.result_window.iconbitmap('logo_image.ico')
        except Exception:
            pass
        
        width = self.result_window.winfo_screenwidth()
        height = self.result_window.winfo_screenheight()
        
        self.result_window.geometry("%dx%d" % (width, height))
        self.result_window.configure(bg="#f0f0f0")
        self.result_window.option_add("*Font", "Helvetica 10")
        self.result_window.option_add("*Button.Background", "#4CAF50")
        self.result_window.option_add("*Button.Foreground", "#ffffff")
        self.result_window.state("zoomed")
        
        
        background_path = 'background_img.jpg' if os.path.exists('background_img.jpg') else None
        if background_path:
            background_image = Image.open(background_path).resize((width, height), Image.LANCZOS)
        else:
            background_image = Image.new('RGB', (width, height), '#07111f')
        self.background_photo = ImageTk.PhotoImage(background_image)
        
        
        self.background_label = tk.Label(self.result_window, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)
        
        
        tk.Label(self.result_window, text="Before encoding", font=('Helvetica', 12)).grid(row=8, column=0, pady=10)
        tk.Label(self.result_window, text="After encoding", font=('Helvetica', 12)).grid(row=8, column=2, pady=10)
    
        
        btn_elbow = tk.Button(self.result_window, text="Elbow Plot", command=self.display_elbow_plot)
        btn_elbow.grid(row=0, column=0, pady=10, padx=10)

        btn_clusters = tk.Button(self.result_window, text="Clusters Plot", command=self.display_clusters_plot)
        btn_clusters.grid(row=0, column=1, pady=10, padx=10)

        btn_cross_clusters_silhouette = tk.Button(self.result_window, text="Cross Clusters Silhouette Plot", command=self.display_cross_clusters_silhouette_plot)
        btn_cross_clusters_silhouette.grid(row=3, column=0, pady=10, padx=10)

        btn_cross_clusters_davies = tk.Button(self.result_window, text="Cross Clusters Davies Bouldin Plot", command=self.display_cross_clusters_davies_plot)
        btn_cross_clusters_davies.grid(row=3, column=1, pady=10, padx=10)

        btn_dendrogram_silhouette = tk.Button(self.result_window, text="dendrogram Silhouette", command=self.display_dendrogram_silhouette)
        btn_dendrogram_silhouette.grid(row=4, column=0, pady=10, padx=10)

        btn_dendrogram_davies = tk.Button(self.result_window, text="dendrogram Davies Bouldin", command=self.display_dendrogram_davies)
        btn_dendrogram_davies.grid(row=4, column=1, pady=10, padx=10)
        
        # No second trajectory-analysis button: the main GUI now runs the single
        # final trajectory workflow directly.
        
        
        self.back_button = tk.Button(self.result_window, text="Back to Clustering 3D Data", command=self.return_to_main, bg="black", fg="white")
        self.back_button.grid(row=7, column=3, columnspan=2, pady=10)
        
        
        self.exit_button = tk.Button(self.result_window, text="Exit", command=self.exit_window, bg="red", fg="white")
        self.exit_button.grid(row=8, column=3, columnspan=2, pady=10)
        
        self.result_window.protocol("WM_DELETE_WINDOW", self.return_to_main)
    
    # ------------------------------------------------------------------
    
    # This is the sole ARI/NMI/PCA/Conv1D-AE validation workflow.
    
    
    # ------------------------------------------------------------------
    def run_trajectory_analysis(self):
        """Run the single final temporally ordered trajectory analysis.

        Raw trajectory features are retained as the mandatory before-AE baseline.
        Conv1D-AE latent features are the after-AE representation. PCA is computed
        only as an auxiliary/supplementary representation and is not a second analysis.
        """
        if self.trajectory_running:
            messagebox.showinfo("Final Trajectory Analysis", "The final trajectory analysis is already running.")
            return
        if not self.file_names or not all(self.file_names):
            messagebox.showwarning("Final Trajectory Analysis", "Please select the input Excel files first.")
            return
        self.trajectory_running = True
        if hasattr(self, "btn_results") and self.btn_results.winfo_exists():
            self.btn_results.config(state="disabled", text="Final Analysis Running...")
        threading.Thread(target=self._trajectory_worker_entry, daemon=True).start()

    def _trajectory_worker_entry(self):
        try:
            folder = self._run_trajectory_analysis_worker()
            self.master.after(0, lambda f=folder: self._trajectory_finished(f))
        except Exception as exc:
            print(traceback.format_exc())
            self.master.after(0, lambda e=str(exc): self._trajectory_failed(e))

    def _trajectory_finished(self, folder):
        self.trajectory_running = False
        if hasattr(self, "btn_results") and self.btn_results.winfo_exists():
            self.btn_results.config(state="normal", text="Run Final Trajectory Analysis")
        messagebox.showinfo(
            "Final Trajectory Analysis",
            "Final trajectory analysis completed.\n\n"
            "The Raw trajectory representation is the before-AE baseline and the "
            "Conv1D-AE representation is the after-AE result.\n\n"
            "Outputs were saved in:\n" + folder +
            "\n\nMain comparison: Main_Results/raw_vs_AE_improvement_summary.csv"
            "\n3D raw clusters: Main_Results/trajectory_clusters_raw_before_AE_3D.png"
            "\n3D AE clusters: Main_Results/trajectory_clusters_after_AE_3D.png"
        )
        self._show_final_trajectory_cluster_results(folder)

    def _show_final_trajectory_cluster_results(self, folder):
        """Display the saved final Raw and After-AE 3D cluster grids in the GUI."""
        main_folder = os.path.join(folder, "Main_Results")
        raw_path = os.path.join(main_folder, "trajectory_clusters_raw_before_AE_3D.png")
        ae_path = os.path.join(main_folder, "trajectory_clusters_after_AE_3D.png")

        if not os.path.exists(raw_path) or not os.path.exists(ae_path):
            messagebox.showwarning(
                "3D Cluster Plots",
                "The analysis finished, but the 3D cluster image files were not found.\n\n"
                f"Raw: {raw_path}\nAE: {ae_path}"
            )
            return

        win = tk.Toplevel(self.master)
        win.title("Final Trajectory 3D Clusters: Before vs After AE")
        try:
            win.state("zoomed")
        except Exception:
            win.geometry("1500x900")
        win.configure(bg="white")

        tk.Label(
            win, text="Final Trajectory-Level 3D Clustering",
            font=("Helvetica", 18, "bold"), bg="white"
        ).pack(pady=(10, 2))
        tk.Label(
            win,
            text="Raw trajectory (before AE) compared with Conv1D-AE (after AE)",
            font=("Helvetica", 11), bg="white"
        ).pack(pady=(0, 8))

        holder = tk.Frame(win, bg="white")
        holder.pack(fill="both", expand=True, padx=10, pady=6)
        holder.grid_columnconfigure(0, weight=1)
        holder.grid_columnconfigure(1, weight=1)

        tk.Label(holder, text="Before AE: Raw trajectory", font=("Helvetica", 13, "bold"), bg="white").grid(row=0, column=0, pady=5)
        tk.Label(holder, text="After AE: Conv1D-AE", font=("Helvetica", 13, "bold"), bg="white").grid(row=0, column=1, pady=5)

        try:
            raw_img = Image.open(raw_path).convert("RGB")
            ae_img = Image.open(ae_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror(
                "3D Cluster Plot Error",
                "The analysis completed successfully, but the saved 3D cluster images could not be loaded.\n\n"
                f"{exc}"
            )
            win.destroy()
            return

        raw_img.thumbnail((700, 700), Image.LANCZOS)
        ae_img.thumbnail((700, 700), Image.LANCZOS)
        raw_tk = ImageTk.PhotoImage(raw_img)
        ae_tk = ImageTk.PhotoImage(ae_img)

        # Keep persistent references on the window so Tkinter does not garbage-collect the images.
        win.raw_tk = raw_tk
        win.ae_tk = ae_tk

        raw_label = tk.Label(holder, image=win.raw_tk, bg="white")
        raw_label.grid(row=1, column=0, padx=8, pady=8, sticky="n")

        ae_label = tk.Label(holder, image=win.ae_tk, bg="white")
        ae_label.grid(row=1, column=1, padx=8, pady=8, sticky="n")

        tk.Button(win, text="Close", command=win.destroy, bg="black", fg="white", padx=28).pack(pady=(4, 12))

    def _trajectory_failed(self, error_message):
        self.trajectory_running = False
        if hasattr(self, "btn_results") and self.btn_results.winfo_exists():
            self.btn_results.config(state="normal", text="Run Final Trajectory Analysis")
        messagebox.showerror("Final Trajectory Analysis", error_message)

    @staticmethod
    def _trajectory_group_name(file_name):
        """Map replicate filenames to the same biological group naming used by the main analysis."""
        import re
        name = os.path.splitext(os.path.basename(file_name))[0]
        low = name.lower().replace('_', ' ').strip()
        if 'male & male' in low or 'male and male' in low:
            return 'Male-Male'
        if 'female & female' in low or 'female and female' in low:
            return 'Female-Female'
        if 'male & female' in low or 'male and female' in low:
            return 'Male-Female'
        for token, label in [('00-01', '0-1 min'), ('05-06', '5-6 min'),
                             ('10-11', '10-11 min'), ('15-16', '15-16 min'),
                             ('20-21', '20-21 min')]:
            if token in low:
                return label
        if 'control' in low:
            return 'Control'
        if '0.25% etoh' in low or '0.25% ethanol' in low:
            return '0.25% Ethanol'
        if ('1% etoh' in low or '1.0% etoh' in low or
                '1% ethanol' in low or '1.0% ethanol' in low):
            return '1.0% Ethanol'
        if 'cat fish' in low or 'catfish' in low:
            return 'Glass Catfish'
        if 'ory combined' in low or 'medaka' in low or 'oryzias' in low:
            return 'Medaka'
        if 'zebrafish' in low:
            return 'Zebrafish'
        if 'unamputated' in low:
            return 'Unamputated'
        if 'caudal fin amputated' in low and ('lidocaine' in low or '5 ppm' in low):
            return 'Caudal Fin Amputated + Lidocaine'
        if low in {'caudal fin', 'caudal fin amputated', 'caudal fin amputation'} or 'caudal fin.xlsx' in low:
            return 'Caudal Fin Amputated'
        if 'lidocaine' in low or '5 ppm lidocaine' in low:
            return 'Lidocaine only'
        clean = re.sub(r'\s*\((?:1st|2nd|3rd|[0-9]+(?:st|nd|rd|th)?)\s+batch\)\s*$', '', name, flags=re.I)
        clean = re.sub(r'\s+(?:0?[1-9]|[1-9][0-9])\s*$', '', clean).strip(' -_')
        return clean

    @staticmethod
    def _temporal_features_from_xyz(xyz):
        """Return frame-ordered [X,Y,Z,speed,acceleration,turning_angle,dZ].

        Speed/acceleration are in coordinate units per frame because the original
        spreadsheets do not provide an explicit frame-time column. No temporal
        rows are randomly removed, preserving within-sheet ordering.
        """
        xyz = np.asarray(xyz, dtype=np.float32)
        velocity = np.zeros_like(xyz, dtype=np.float32)
        velocity[1:] = np.diff(xyz, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        acceleration = np.zeros(len(xyz), dtype=np.float32)
        acceleration[1:] = np.diff(speed)

        turning = np.zeros(len(xyz), dtype=np.float32)
        if len(xyz) >= 3:
            v1 = velocity[1:-1]
            v2 = velocity[2:]
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cosang = np.ones(len(v1), dtype=np.float32)
            valid = denom > 1e-12
            cosang[valid] = np.sum(v1[valid] * v2[valid], axis=1) / denom[valid]
            cosang = np.clip(cosang, -1.0, 1.0)
            turning[2:] = np.arccos(cosang)

        dz = velocity[:, 2]
        return np.column_stack([xyz, speed, acceleration, turning, dz]).astype(np.float32)

    @staticmethod
    def _make_windows(features, window_size, stride):
        """Create fixed-length continuous windows without crossing trajectory boundaries."""
        if len(features) < window_size:
            return np.empty((0, window_size, features.shape[1]), dtype=np.float32), []
        starts = list(range(0, len(features) - window_size + 1, stride))
        windows = np.stack([features[s:s + window_size] for s in starts]).astype(np.float32)
        return windows, starts

    @staticmethod
    def _build_trajectory_autoencoder(window_size, n_features, code_size=8):
        """Build a stronger temporal Conv1D denoising autoencoder.

        The encoder deliberately avoids GlobalAveragePooling1D because averaging
        across the full window can erase short-lived locomotor structure. The
        default bottleneck is 8D rather than 3D so the 100 x 7 trajectory window
        is not compressed too aggressively. Biological/group labels are never
        used to train this network.
        """
        inp = keras.layers.Input(shape=(window_size, n_features), name="trajectory_window")
        x = keras.layers.GaussianNoise(0.05, name="denoising_noise")(inp)
        x = keras.layers.Conv1D(32, 5, activation="relu", padding="same", name="enc_conv1")(x)
        x = keras.layers.BatchNormalization(name="enc_bn1")(x)
        x = keras.layers.Conv1D(64, 5, activation="relu", padding="same", name="enc_conv2")(x)
        x = keras.layers.BatchNormalization(name="enc_bn2")(x)
        x = keras.layers.Conv1D(128, 3, activation="relu", padding="same", name="enc_conv3")(x)
        x = keras.layers.BatchNormalization(name="enc_bn3")(x)
        x = keras.layers.Flatten(name="enc_flatten")(x)
        x = keras.layers.Dense(64, activation="relu", name="enc_dense")(x)
        code = keras.layers.Dense(code_size, activation="linear", name="trajectory_latent")(x)
        encoder = keras.models.Model(inp, code, name="trajectory_encoder")

        code_in = keras.layers.Input(shape=(code_size,), name="trajectory_code")
        y = keras.layers.Dense(window_size * 64, activation="relu", name="dec_dense")(code_in)
        y = keras.layers.Reshape((window_size, 64), name="dec_reshape")(y)
        y = keras.layers.Conv1DTranspose(128, 3, activation="relu", padding="same", name="dec_deconv1")(y)
        y = keras.layers.BatchNormalization(name="dec_bn1")(y)
        y = keras.layers.Conv1DTranspose(64, 5, activation="relu", padding="same", name="dec_deconv2")(y)
        y = keras.layers.BatchNormalization(name="dec_bn2")(y)
        y = keras.layers.Conv1DTranspose(32, 5, activation="relu", padding="same", name="dec_deconv3")(y)
        y = keras.layers.BatchNormalization(name="dec_bn3")(y)
        recon = keras.layers.Conv1D(n_features, 3, activation="linear", padding="same", name="trajectory_reconstruction")(y)
        decoder = keras.models.Model(code_in, recon, name="trajectory_decoder")

        auto_in = keras.layers.Input(shape=(window_size, n_features), name="trajectory_autoencoder_input")
        auto_out = decoder(encoder(auto_in))
        autoencoder = keras.models.Model(auto_in, auto_out, name="trajectory_conv1d_autoencoder")
        return encoder, decoder, autoencoder

    @staticmethod
    def _trajectory_student_t_assignments(z, centers, alpha=1.0):
        """Student-t soft cluster assignments used for unsupervised DEC-style refinement."""
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        centers = tf.convert_to_tensor(centers, dtype=tf.float32)
        dist2 = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - tf.expand_dims(centers, 0)), axis=2)
        q = 1.0 / (1.0 + dist2 / alpha)
        q = tf.pow(q, (alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    @staticmethod
    def _trajectory_target_distribution(q):
        """DEC target distribution that sharpens confident unsupervised assignments."""
        q = tf.convert_to_tensor(q, dtype=tf.float32)
        weight = tf.square(q) / (tf.reduce_sum(q, axis=0, keepdims=True) + 1e-8)
        return weight / (tf.reduce_sum(weight, axis=1, keepdims=True) + 1e-8)

    def _refine_trajectory_autoencoder_for_clustering(
            self, encoder, decoder, train_windows, n_clusters,
            epochs=40, batch_size=256, cluster_weight=0.10, seed=55):
        """Unsupervised clustering-aware refinement using only training recordings.

        Cluster centers are initialized from K-Means on the current latent codes.
        The refinement minimizes reconstruction MSE plus a DEC-style KL term.
        Experimental/group labels and ARI/NMI are not used for optimization or
        architecture selection, so external agreement remains a post hoc test.
        """
        if len(train_windows) < max(4, n_clusters):
            return pd.DataFrame(columns=['Epoch', 'Total_Loss', 'Reconstruction_Loss', 'Clustering_KL'])

        z0 = encoder.predict(train_windows, batch_size=256, verbose=0)
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        km.fit(z0)
        centers = tf.Variable(km.cluster_centers_.astype(np.float32), trainable=True, name='latent_cluster_centers')

        optimizer = Adam(learning_rate=2e-5)
        train_tensor = tf.convert_to_tensor(train_windows, dtype=tf.float32)
        n = len(train_windows)
        rng = np.random.default_rng(seed)
        history_rows = []

        for epoch in range(1, int(epochs) + 1):
            # Update the sharpened target distribution once per epoch using all
            # training windows. This remains fully unsupervised.
            z_all = encoder(train_tensor, training=False)
            q_all = self._trajectory_student_t_assignments(z_all, centers)
            p_all = tf.stop_gradient(self._trajectory_target_distribution(q_all))

            order = rng.permutation(n)
            total_vals, recon_vals, kl_vals = [], [], []
            for start in range(0, n, int(batch_size)):
                ids = order[start:start + int(batch_size)]
                xb = tf.gather(train_tensor, ids)
                pb = tf.gather(p_all, ids)
                with tf.GradientTape() as tape:
                    zb = encoder(xb, training=True)
                    rb = decoder(zb, training=True)
                    qb = self._trajectory_student_t_assignments(zb, centers)
                    recon_loss = tf.reduce_mean(tf.square(xb - rb))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(
                        pb * tf.math.log((pb + 1e-8) / (qb + 1e-8)), axis=1
                    ))
                    total_loss = recon_loss + float(cluster_weight) * kl_loss

                variables = encoder.trainable_variables + decoder.trainable_variables + [centers]
                grads = tape.gradient(total_loss, variables)
                optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
                total_vals.append(float(total_loss.numpy()))
                recon_vals.append(float(recon_loss.numpy()))
                kl_vals.append(float(kl_loss.numpy()))

            history_rows.append({
                'Epoch': epoch,
                'Total_Loss': float(np.mean(total_vals)),
                'Reconstruction_Loss': float(np.mean(recon_vals)),
                'Clustering_KL': float(np.mean(kl_vals)),
            })
            if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
                print(
                    f"[F3DCA] AE clustering refinement {epoch}/{epochs}: "
                    f"total={history_rows[-1]['Total_Loss']:.5f}, "
                    f"recon={history_rows[-1]['Reconstruction_Loss']:.5f}, "
                    f"KL={history_rows[-1]['Clustering_KL']:.5f}",
                    flush=True
                )

        return pd.DataFrame(history_rows)

    def _refine_trajectory_autoencoder_for_gmm(
            self, encoder, decoder, train_windows, n_clusters,
            epochs=30, batch_size=256, cluster_weight=0.08, compact_weight=0.02, seed=55):
        """Unsupervised GMM-aware latent refinement using training recordings only.

        A GaussianMixture model is fitted to the reconstruction-pretrained latent
        codes. Its soft responsibilities are used as pseudo-targets; no biological
        labels, ARI, or NMI enter optimization. The encoder/decoder are then
        refined to preserve reconstruction while making the latent geometry more
        compatible with mixture components.
        """
        if len(train_windows) < max(8, n_clusters * 2):
            return pd.DataFrame(columns=['Epoch','Total_Loss','Reconstruction_Loss','GMM_KL','Compactness'])

        z0 = encoder.predict(train_windows, batch_size=256, verbose=0)

        # Keep the original GMM settings first so datasets that already run
        # successfully produce the same results as before. Only if sklearn
        # reports an ill-defined covariance do we retry with progressively
        # stronger covariance regularisation. This fallback is dataset-local
        # and is activated only on failure (e.g. a collapsed ethanol latent
        # component); it does not change settings for successful datasets.
        gmm = None
        last_gmm_error = None
        for reg in (1e-5, 1e-4, 1e-3, 1e-2):
            candidate = GaussianMixture(
                n_components=n_clusters, covariance_type='diag', n_init=20,
                reg_covar=reg, random_state=seed
            )
            try:
                candidate.fit(z0)
                gmm = candidate
                if reg != 1e-5:
                    print(
                        f"[F3DCA] GMM refinement covariance fallback used: "
                        f"reg_covar={reg:g}",
                        flush=True
                    )
                break
            except ValueError as exc:
                last_gmm_error = exc
                msg = str(exc).lower()
                if ('covariance' not in msg and 'ill-defined' not in msg and
                        'positive definite' not in msg):
                    raise

        if gmm is None:
            raise ValueError(
                "GMM-aware refinement failed even after covariance "
                "regularisation retries (1e-5 to 1e-2)."
            ) from last_gmm_error

        resp = gmm.predict_proba(z0).astype(np.float32)
        centers = tf.Variable(gmm.means_.astype(np.float32), trainable=True, name='gmm_latent_centers')
        log_scales = tf.Variable(
            np.log(np.sqrt(gmm.covariances_ + 1e-5)).astype(np.float32),
            trainable=True, name='gmm_log_scales'
        )

        optimizer = Adam(learning_rate=1e-5)
        train_tensor = tf.convert_to_tensor(train_windows, dtype=tf.float32)
        resp_tensor = tf.convert_to_tensor(resp, dtype=tf.float32)
        n = len(train_windows)
        rng = np.random.default_rng(seed)
        history_rows = []

        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(n)
            total_vals, recon_vals, kl_vals, comp_vals = [], [], [], []
            for start in range(0, n, int(batch_size)):
                ids = order[start:start + int(batch_size)]
                xb = tf.gather(train_tensor, ids)
                rb_target = tf.gather(resp_tensor, ids)
                with tf.GradientTape() as tape:
                    zb = encoder(xb, training=True)
                    recon = decoder(zb, training=True)
                    recon_loss = tf.reduce_mean(tf.square(xb - recon))

                    scales = tf.nn.softplus(log_scales) + 1e-4
                    diff = tf.expand_dims(zb, 1) - tf.expand_dims(centers, 0)
                    mahal = tf.reduce_sum(
                        tf.square(diff) / tf.expand_dims(tf.square(scales), 0), axis=2
                    )
                    log_det = tf.reduce_sum(tf.math.log(tf.square(scales)), axis=1)
                    logits = -0.5 * (mahal + tf.expand_dims(log_det, 0))
                    q = tf.nn.softmax(logits, axis=1)
                    gmm_kl = tf.reduce_mean(tf.reduce_sum(
                        rb_target * tf.math.log((rb_target + 1e-8) / (q + 1e-8)), axis=1
                    ))
                    compact = tf.reduce_mean(tf.reduce_sum(
                        tf.expand_dims(rb_target, 2) * tf.square(diff), axis=[1,2]
                    ))
                    total_loss = recon_loss + float(cluster_weight) * gmm_kl + float(compact_weight) * compact

                variables = encoder.trainable_variables + decoder.trainable_variables + [centers, log_scales]
                grads = tape.gradient(total_loss, variables)
                optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
                total_vals.append(float(total_loss.numpy()))
                recon_vals.append(float(recon_loss.numpy()))
                kl_vals.append(float(gmm_kl.numpy()))
                comp_vals.append(float(compact.numpy()))

            history_rows.append({
                'Epoch': epoch,
                'Total_Loss': float(np.mean(total_vals)),
                'Reconstruction_Loss': float(np.mean(recon_vals)),
                'GMM_KL': float(np.mean(kl_vals)),
                'Compactness': float(np.mean(comp_vals)),
            })
            if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
                print(
                    f"[F3DCA] GMM-aware AE refinement {epoch}/{epochs}: "
                    f"total={history_rows[-1]['Total_Loss']:.5f}, "
                    f"recon={history_rows[-1]['Reconstruction_Loss']:.5f}, "
                    f"KL={history_rows[-1]['GMM_KL']:.5f}, "
                    f"compact={history_rows[-1]['Compactness']:.5f}",
                    flush=True
                )

        return pd.DataFrame(history_rows)

    def _load_trajectory_windows(self, window_size=100, stride=None):
        """Treat each Excel sheet as one ordered trajectory/recording unit."""
        if stride is None:
            stride = max(1, window_size // 2)
        windows, metadata = [], []
        feature_names = ['X', 'Y', 'Z', 'Speed', 'Acceleration', 'TurningAngle', 'dZ']

        for file_index, file_name in enumerate(self.file_names):
            group_name = self._trajectory_group_name(file_name)
            xls = pd.ExcelFile(file_name)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                if df.shape[1] < 3:
                    continue
                xyz_df = df.iloc[:, :3].copy()
                xyz_df.columns = ['X', 'Y', 'Z']
                for c in ['X', 'Y', 'Z']:
                    xyz_df[c] = pd.to_numeric(xyz_df[c], errors='coerce')
                # Interpolate missing coordinates within the same trajectory instead
                # of deleting random temporal observations.
                xyz_df = xyz_df.interpolate(limit_direction='both').dropna()
                if len(xyz_df) < window_size:
                    continue
                feat = self._temporal_features_from_xyz(xyz_df[['X', 'Y', 'Z']].to_numpy())
                sheet_windows, starts = self._make_windows(feat, window_size, stride)
                for win, start in zip(sheet_windows, starts):
                    windows.append(win)
                    metadata.append({
                        'Group': group_name,
                        'File': os.path.basename(file_name),
                        'File_Index': file_index,
                        'Sheet': str(sheet_name),
                        'Trajectory_ID': f"file{file_index}::{sheet_name}",
                        'Start_Row': int(start),
                        'End_Row': int(start + window_size - 1),
                        'Window_Size': int(window_size),
                        'Stride': int(stride),
                    })
        if not windows:
            return np.empty((0, window_size, len(feature_names)), dtype=np.float32), pd.DataFrame(), feature_names
        return np.stack(windows).astype(np.float32), pd.DataFrame(metadata), feature_names


    @staticmethod
    def _trajectory_train_val_split(metadata, val_fraction=0.20, seed=55):
        """Split whole recording/Trajectory_ID units, never overlapping windows.

        The split is performed within each biological group when at least two
        independent trajectory IDs are available. A trajectory ID is assigned
        wholly to either training or validation, so overlapping windows from the
        same recording can never cross partitions. Groups represented by only
        one trajectory are retained in training and are flagged in the saved
        partition table as not independently validated.
        """
        if metadata.empty or 'Trajectory_ID' not in metadata.columns:
            raise ValueError('Trajectory metadata with Trajectory_ID is required for grouped splitting.')

        rng = np.random.default_rng(seed)
        train_ids, val_ids = [], []
        for group_name, group_meta in metadata.groupby('Group', sort=False):
            tids = np.asarray(pd.unique(group_meta['Trajectory_ID']), dtype=object)
            tids = np.sort(tids.astype(str))
            if len(tids) < 2:
                train_ids.extend(tids.tolist())
                continue
            shuffled = tids.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(round(val_fraction * len(shuffled))))
            n_val = min(n_val, len(shuffled) - 1)
            val_ids.extend(shuffled[:n_val].tolist())
            train_ids.extend(shuffled[n_val:].tolist())

        # If every group has only one trajectory, an independent validation split
        # is impossible. Keep all data in training and report this explicitly.
        train_ids = set(train_ids)
        val_ids = set(val_ids)
        if not train_ids:
            train_ids = set(pd.unique(metadata['Trajectory_ID']).astype(str))

        tids_series = metadata['Trajectory_ID'].astype(str)
        train_idx = np.flatnonzero(tids_series.isin(train_ids).to_numpy())
        val_idx = np.flatnonzero(tids_series.isin(val_ids).to_numpy())

        partition = metadata[['Group', 'File', 'Sheet', 'Trajectory_ID']].drop_duplicates().copy()
        partition['Partition'] = np.where(
            partition['Trajectory_ID'].astype(str).isin(val_ids), 'Validation', 'Training'
        )
        partition['Independent_Validation_Available_For_Group'] = partition.groupby('Group')['Partition'].transform(
            lambda x: bool((x == 'Validation').any() and (x == 'Training').any())
        )
        return train_idx, val_idx, partition

    def _trajectory_recording_permutation_null(self, true_labels, predicted_labels, metadata, eval_idx, n_permutations=1000, seed=55):
        """Recording-level permutation null for ARI.

        Biological labels are shuffled across independent Trajectory_ID units,
        then propagated to all windows from each trajectory. This preserves the
        dependence among overlapping windows and avoids an invalid window-level
        permutation.
        """
        eval_idx = np.asarray(eval_idx, dtype=int)
        meta_eval = metadata.iloc[eval_idx].reset_index(drop=True)
        observed = adjusted_rand_score(np.asarray(true_labels)[eval_idx], predicted_labels)

        rec = meta_eval[['Trajectory_ID', 'Group']].drop_duplicates('Trajectory_ID').reset_index(drop=True)
        trajectory_ids = rec['Trajectory_ID'].astype(str).to_numpy()
        trajectory_groups = rec['Group'].astype(str).to_numpy()
        if len(trajectory_ids) < 2:
            return observed, np.array([], dtype=float), np.nan, np.nan, np.nan, np.nan

        rng = np.random.default_rng(seed)
        null = np.empty(int(n_permutations), dtype=float)
        window_tid = meta_eval['Trajectory_ID'].astype(str).to_numpy()
        for i in range(int(n_permutations)):
            shuffled_groups = rng.permutation(trajectory_groups)
            perm_map = dict(zip(trajectory_ids, shuffled_groups))
            perm_group_strings = np.array([perm_map[t] for t in window_tid], dtype=object)
            # Convert permuted group strings to integer labels without using any
            # information from the predicted clustering.
            _, perm_labels = np.unique(perm_group_strings, return_inverse=True)
            null[i] = adjusted_rand_score(perm_labels, predicted_labels)

        lower, upper = np.quantile(null, [0.025, 0.975])
        # One-sided test: is observed agreement greater than expected by chance?
        p_value = (1.0 + np.sum(null >= observed)) / (len(null) + 1.0)
        return observed, null, float(np.mean(null)), float(lower), float(upper), float(p_value)

    def _evaluate_trajectory_representations(self, raw_flat, pca_data, encoded_data, labels, metadata, n_clusters, n_permutations=1000):
        """Primary cross-representation evaluation using external ARI/NMI.

        Silhouette and Davies-Bouldin are retained only as within-representation
        geometric diagnostics; they must not be interpreted as evidence that one
        representation is biologically superior to another.
        """
        eval_cap = min(3000, len(labels))
        rng = np.random.default_rng(55)
        if len(labels) > eval_cap:
            idx = np.sort(rng.choice(len(labels), eval_cap, replace=False))
        else:
            idx = np.arange(len(labels))

        models_factory = lambda: [
            ('KMeans', KMeans(n_clusters=n_clusters, n_init=10, random_state=55)),
            ('Birch', Birch(n_clusters=n_clusters, threshold=0.5)),
            ('GMM', GaussianMixture(n_components=n_clusters, n_init=5, random_state=55)),
            ('Spectral', SpectralClustering(
                n_clusters=n_clusters,
                n_init=10,
                random_state=55,
                affinity='nearest_neighbors',
                n_neighbors=15
            )),
        ]
        rows = []
        predictions = {}
        permutation_rows = []
        for rep_name, data in [('Raw trajectory', raw_flat), ('PCA', pca_data), ('Conv1D-AE', encoded_data)]:
            d = np.asarray(data)[idx]
            lab = np.asarray(labels)[idx]
            for method_name, model in models_factory():
                pred = model.fit_predict(d)
                ari, null, null_mean, null_low, null_high, perm_p = self._trajectory_recording_permutation_null(
                    labels, pred, metadata, idx, n_permutations=n_permutations,
                    seed=55 + len(rows)
                )
                rows.append({
                    'Representation': rep_name,
                    'Method': method_name,
                    'Evaluation_N_Windows': len(d),
                    'Evaluation_N_Trajectory_IDs': metadata.iloc[idx]['Trajectory_ID'].nunique(),
                    'ARI': ari,
                    'NMI': normalized_mutual_info_score(lab, pred),
                    'Permutation_N': len(null),
                    'Null_ARI_Mean': null_mean,
                    'Null_ARI_2.5%': null_low,
                    'Null_ARI_97.5%': null_high,
                    'ARI_Permutation_P_OneSided': perm_p,
                    'Silhouette_within_representation': silhouette_score(d, pred),
                    'Davies_Bouldin_within_representation': davies_bouldin_score(d, pred),
                })
                if len(null):
                    for perm_i, value in enumerate(null, start=1):
                        permutation_rows.append({
                            'Representation': rep_name, 'Method': method_name,
                            'Permutation': perm_i, 'Null_ARI': value
                        })
                predictions[(rep_name, method_name)] = (idx, pred)
        return pd.DataFrame(rows), predictions, pd.DataFrame(permutation_rows)

    def _groupwise_internal_metrics_from_global_clustering(self, raw_flat, encoded_data, metadata, predictions, output_folder):
        """Export post-hoc groupwise Silhouette/DB without refitting clustering.

        IMPORTANT: this function does NOT change the clustering, autoencoder,
        train/validation split, ARI/NMI, or any dataset-level score. It simply
        takes the already-fitted global clustering assignments and calculates
        Silhouette and Davies-Bouldin diagnostics within each experimental
        group. This preserves the revised trajectory-level workflow while also
        providing the separated group rows used in the manuscript tables.
        """
        representation_data = {
            'Raw trajectory': np.asarray(raw_flat),
            'Conv1D-AE': np.asarray(encoded_data),
        }
        method_order = ['KMeans', 'Birch', 'GMM', 'Spectral']
        group_order = list(pd.unique(metadata['Group'].astype(str)))
        rows = []

        for rep_name in ['Raw trajectory', 'Conv1D-AE']:
            data_all = representation_data[rep_name]
            for method_name in method_order:
                key = (rep_name, method_name)
                if key not in predictions:
                    continue
                eval_idx, pred = predictions[key]
                eval_idx = np.asarray(eval_idx, dtype=int)
                pred = np.asarray(pred)
                eval_meta = metadata.iloc[eval_idx].reset_index(drop=True)
                eval_data = data_all[eval_idx]

                for group_name in group_order:
                    mask = (eval_meta['Group'].astype(str).to_numpy() == str(group_name))
                    group_data = eval_data[mask]
                    group_pred = pred[mask]
                    unique_clusters = np.unique(group_pred)
                    n_group = int(mask.sum())
                    n_present = int(len(unique_clusters))

                    sil = np.nan
                    db = np.nan
                    status = 'OK'
                    # Silhouette requires 2 <= number of labels <= n_samples-1.
                    # DB also requires at least two predicted clusters.
                    if n_group >= 3 and 2 <= n_present < n_group:
                        try:
                            sil = float(silhouette_score(group_data, group_pred))
                            db = float(davies_bouldin_score(group_data, group_pred))
                        except Exception as exc:
                            status = f'Not computable: {exc}'
                    else:
                        status = f'Not computable: {n_present} predicted cluster(s) in {n_group} windows'

                    rows.append({
                        'Group': group_name,
                        'Representation': rep_name,
                        'Method': method_name,
                        'N_Windows': n_group,
                        'N_Predicted_Clusters_Present': n_present,
                        'Silhouette': sil,
                        'Davies_Bouldin': db,
                        'Status': status,
                    })

        long_df = pd.DataFrame(rows)
        long_path = os.path.join(output_folder, 'groupwise_internal_metrics_from_global_clustering_long.csv')
        long_df.to_csv(long_path, index=False)

        
        raw_df = long_df[long_df['Representation'] == 'Raw trajectory'].set_index(['Group', 'Method'])
        ae_df = long_df[long_df['Representation'] == 'Conv1D-AE'].set_index(['Group', 'Method'])
        wide_rows = []
        for group_name in group_order:
            for method_name in method_order:
                key = (group_name, method_name)
                if key not in raw_df.index or key not in ae_df.index:
                    continue
                wide_rows.append({
                    'Group': group_name,
                    'Method': method_name,
                    'Silhouette_Before': raw_df.loc[key, 'Silhouette'],
                    'Silhouette_After': ae_df.loc[key, 'Silhouette'],
                    'DB_Before': raw_df.loc[key, 'Davies_Bouldin'],
                    'DB_After': ae_df.loc[key, 'Davies_Bouldin'],
                    'N_Windows_Before': int(raw_df.loc[key, 'N_Windows']),
                    'N_Windows_After': int(ae_df.loc[key, 'N_Windows']),
                    'Raw_Status': raw_df.loc[key, 'Status'],
                    'AE_Status': ae_df.loc[key, 'Status'],
                })
        wide_df = pd.DataFrame(wide_rows)
        wide_path = os.path.join(output_folder, 'groupwise_before_vs_after_AE_internal_metrics.csv')
        wide_df.to_csv(wide_path, index=False)
        return long_df, wide_df

    def _project_representation_to_3d(self, data, train_idx=None):
        """Project a representation to 3D for visualization only.
        Clustering itself is performed in the original/full representation.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            arr = arr.reshape(len(arr), -1)
        if arr.shape[1] == 3:
            return arr, ['Dim 1', 'Dim 2', 'Dim 3']
        if arr.shape[1] < 3:
            out = np.zeros((arr.shape[0], 3), dtype=float)
            out[:, :arr.shape[1]] = arr
            return out, ['Dim 1', 'Dim 2', 'Dim 3']
        pca3 = PCA(n_components=3, random_state=55)
        if train_idx is not None and len(train_idx):
            fit_idx = np.asarray(train_idx)
            pca3.fit(arr[fit_idx])
        else:
            pca3.fit(arr)
        proj = pca3.transform(arr)
        labels = ['PC1', 'PC2', 'PC3']
        return proj, labels

    def _save_trajectory_3d_cluster_grid(self, data, representation_name, predictions, metadata, output_path, train_idx=None, individual_dir=None):
        """Save combined and individual 3D cluster plots with clear, publication-style text.
        PCA is used only to project high-dimensional representations to 3D for display;
        clustering predictions are unchanged and come from the full representation.
        """
        proj, axis_labels = self._project_representation_to_3d(data, train_idx=train_idx)
        methods_order = ['KMeans', 'Birch', 'GMM', 'Spectral']
        metadata = metadata.reset_index(drop=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if individual_dir is not None:
            os.makedirs(individual_dir, exist_ok=True)

        rep_title = ('Raw trajectory (before AE)' if representation_name == 'Raw trajectory'
                     else 'Conv1D-AE latent representation (after AE)' if representation_name == 'Conv1D-AE'
                     else representation_name)

        def _clean_group_label(name):
            txt = str(name)
            txt = txt.replace('Female-Female', 'Female–Female')
            txt = txt.replace('Male-Female', 'Male–Female')
            txt = txt.replace('Male-Male', 'Male–Male')
            txt = txt.replace('Caudal Fin Amputated + Lidocaine', 'Amputated + Lidocaine')
            txt = txt.replace('Caudal Fin Amputated', 'Caudal Fin Amputated')
            txt = txt.replace('0-1 min', '0–1 min').replace('5-6 min', '5–6 min')
            txt = txt.replace('10-11 min', '10–11 min').replace('15-16 min', '15–16 min').replace('20-21 min', '20–21 min')
            return txt

        def draw_method(ax, method_name):
            key = (representation_name, method_name)
            if key not in predictions:
                ax.text2D(0.5, 0.5, f'No prediction: {method_name}', transform=ax.transAxes,
                          ha='center', va='center', fontsize=14, fontweight='bold')
                return False
            idx, pred = predictions[key]
            idx = np.asarray(idx, dtype=int)
            pred = np.asarray(pred)
            coords = proj[idx]
            meta_sub = metadata.iloc[idx].reset_index(drop=True)
            cluster_ids = np.unique(pred)
            cmap = plt.get_cmap('tab10', max(len(cluster_ids), 4))

            handles = []
            labels = []
            for c_i, cluster_id in enumerate(cluster_ids):
                mask = pred == cluster_id
                if not np.any(mask):
                    continue
                groups_here = meta_sub.loc[mask, 'Group'] if 'Group' in meta_sub.columns else pd.Series(dtype=object)
                majority_group = groups_here.value_counts().idxmax() if len(groups_here) else f'Cluster {cluster_id}'
                short_group = _clean_group_label(majority_group)
                scatter = ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                                     s=24, alpha=0.82, color=cmap(c_i), edgecolors='none')
                centroid = coords[mask].mean(axis=0)
                ax.scatter(centroid[0], centroid[1], centroid[2], s=180, marker='X',
                           color=cmap(c_i), edgecolor='black', linewidth=1.2)
                handles.append(scatter)
                labels.append(f'Cluster {cluster_id}: {short_group}')

            ax.set_title(method_name, fontsize=18, fontweight='bold', pad=8)
            ax.set_xlabel(axis_labels[0], fontsize=14, fontweight='bold', labelpad=8)
            ax.set_ylabel(axis_labels[1], fontsize=14, fontweight='bold', labelpad=8)
            ax.set_zlabel('')
            ax.text2D(-0.07, 0.50, axis_labels[2], transform=ax.transAxes,
                      rotation=90, fontsize=14, fontweight='bold', va='center', ha='center')
            ax.tick_params(axis='both', which='major', labelsize=10, pad=2)
            for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                tick.set_fontweight('bold')
            ax.view_init(elev=24, azim=42)
            try:
                ax.set_box_aspect((1.10, 1.0, 0.82))
            except Exception:
                pass
            leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                            fontsize=10, frameon=True, ncol=1, borderaxespad=0.20,
                            handletextpad=0.5, labelspacing=0.35, markerscale=1.0)
            leg.get_frame().set_alpha(0.96)
            for txt in leg.get_texts():
                txt.set_fontweight('bold')
            return True

        # Combined 2x2 grid.
        fig = plt.figure(figsize=(20, 14), constrained_layout=False)
        for plot_i, method_name in enumerate(methods_order, start=1):
            ax = fig.add_subplot(2, 2, plot_i, projection='3d')
            draw_method(ax, method_name)
        fig.suptitle(f'3D cluster visualization: {rep_title}\n(3D projection for visualization only)',
                     fontsize=22, fontweight='bold', y=0.96)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.83, wspace=0.18, hspace=0.32)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)
        print(f'[F3DCA] SAVED 3D GRID: {os.path.abspath(output_path)}', flush=True)

        # One PNG per algorithm.
        if individual_dir is not None:
            safe_rep = 'Raw_Before_AE' if representation_name == 'Raw trajectory' else ('AE_After' if representation_name == 'Conv1D-AE' else representation_name.replace(' ', '_'))
            for method_name in methods_order:
                fig = plt.figure(figsize=(11, 8.5), constrained_layout=False)
                ax = fig.add_subplot(111, projection='3d')
                draw_method(ax, method_name)
                fig.suptitle(f'{method_name}: {rep_title}', fontsize=18, fontweight='bold', y=0.95)
                fig.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.84)
                out = os.path.join(individual_dir, f'{safe_rep}_{method_name}_3D.png')
                fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.35)
                plt.close(fig)
                print(f'[F3DCA] SAVED 3D IMAGE: {os.path.abspath(out)}', flush=True)

    def _save_trajectory_elbow_plots(self, raw_data, ae_data, output_dir):
        """Generate descriptive elbow plots for the final trajectory-level analysis.
        These plots are NOT used to choose k; k remains predefined by the number
        of experimental groups. The elbow is shown only for raw trajectory and
        post-AE latent representations.
        """
        os.makedirs(output_dir, exist_ok=True)

        def compute_wcss(data):
            arr = np.asarray(data)
            if arr.ndim != 2:
                arr = arr.reshape(len(arr), -1)
            max_k = min(10, max(2, arr.shape[0]))
            ks = list(range(1, max_k + 1))
            wcss = []
            for k in ks:
                model = KMeans(n_clusters=k, n_init=10, random_state=55)
                model.fit(arr)
                wcss.append(float(model.inertia_))
            return ks, wcss

        raw_ks, raw_wcss = compute_wcss(raw_data)
        ae_ks, ae_wcss = compute_wcss(ae_data)

        # Combined two-panel figure.
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].plot(raw_ks, raw_wcss, marker='o')
        axes[0].set_title('A. Raw trajectory representation', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Within-cluster sum of squares (WCSS)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(ae_ks, ae_wcss, marker='o')
        axes[1].set_title('B. Autoencoder latent representation', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Within-cluster sum of squares (WCSS)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        fig.suptitle('Elbow-method visualization for trajectory-level clustering\n(Descriptive only; k was predefined by experimental group number)', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        combined_path = os.path.join(output_dir, 'trajectory_elbow_raw_vs_AE.png')
        fig.savefig(combined_path, dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)
        print(f'[F3DCA] SAVED ELBOW FIGURE: {os.path.abspath(combined_path)}', flush=True)

        # Individual figures as well.
        for title, ks, wcss, fname in [
            ('Raw trajectory representation', raw_ks, raw_wcss, 'trajectory_elbow_raw.png'),
            ('Autoencoder latent representation', ae_ks, ae_wcss, 'trajectory_elbow_after_AE.png')
        ]:
            plt.figure(figsize=(8, 6))
            plt.plot(ks, wcss, marker='o')
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Number of clusters (k)', fontsize=13, fontweight='bold')
            plt.ylabel('Within-cluster sum of squares (WCSS)', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)
            out = os.path.join(output_dir, fname)
            plt.tight_layout()
            plt.savefig(out, dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close()
            print(f'[F3DCA] SAVED ELBOW FIGURE: {os.path.abspath(out)}', flush=True)

        elbow_df = pd.DataFrame({
            'k': pd.Series(raw_ks),
            'Raw_Trajectory_WCSS': pd.Series(raw_wcss),
            'AE_Latent_WCSS': pd.Series(ae_wcss)
        })
        elbow_df.to_csv(os.path.join(output_dir, 'trajectory_elbow_WCSS_values.csv'), index=False)

    def _run_trajectory_analysis_worker(self):
        np.random.seed(55)
        random.seed(55)
        tf.keras.utils.set_random_seed(55)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        root_folder = os.path.join(self.plot_folder, "Trajectory_Level_Analysis")
        main_folder = os.path.join(root_folder, "Main_Results")
        supp_folder = os.path.join(root_folder, "Supplementary")
        cluster_folder = os.path.join(main_folder, "3D_Clusters")
        os.makedirs(main_folder, exist_ok=True)
        os.makedirs(supp_folder, exist_ok=True)
        os.makedirs(cluster_folder, exist_ok=True)
        print(f'[F3DCA] 3D cluster output folder: {os.path.abspath(cluster_folder)}', flush=True)

        # Primary analysis: 100-frame windows with 50% overlap. If the supplied
        # trajectories are shorter, select the largest prespecified viable window.
        candidate_primary = [100, 50, 25, 10]
        windows = metadata = feature_names = None
        primary_window = None
        for w in candidate_primary:
            temp_windows, temp_meta, temp_names = self._load_trajectory_windows(w, max(1, w // 2))
            if len(temp_windows) >= max(10, len(pd.unique(temp_meta['Group'])) * 2 if len(temp_meta) else 10):
                windows, metadata, feature_names = temp_windows, temp_meta, temp_names
                primary_window = w
                break
        if primary_window is None or len(windows) < 4:
            raise ValueError(
                "Not enough continuous rows within Excel sheets to build trajectory windows. "
                "Each sheet must contain a time-ordered XYZ trajectory with at least 10 rows."
            )

        groups = list(pd.unique(metadata['Group']))
        n_clusters = len(groups)
        if n_clusters < 2:
            raise ValueError("Trajectory analysis requires at least two biological/experimental groups.")
        if len(windows) <= n_clusters:
            raise ValueError("Not enough trajectory windows for the number of biological groups.")
        group_to_id = {g: i for i, g in enumerate(groups)}
        true_labels = metadata['Group'].map(group_to_id).to_numpy(dtype=int)

        # ------------------------------------------------------------------
        # Leakage-safe partitioning at the independent recording level.
        # Split Trajectory_IDs BEFORE fitting any scaler, PCA or autoencoder.
        # Because all overlapping windows from one recording share the same
        # Trajectory_ID, no temporal content can cross train/validation partitions.
        # ------------------------------------------------------------------
        n_windows, window_size, n_features = windows.shape
        train_idx, val_idx, partition_table = self._trajectory_train_val_split(
            metadata, val_fraction=0.20, seed=55
        )
        if len(train_idx) == 0:
            raise ValueError('No trajectory windows were assigned to the training partition.')

        # Fit feature scaler ONLY on frames originating from training recordings.
        scaler = StandardScaler()
        train_frames = windows[train_idx].reshape(-1, n_features)
        scaler.fit(train_frames)
        scaled_frames_all = scaler.transform(windows.reshape(-1, n_features))
        scaled_windows = scaled_frames_all.reshape(n_windows, window_size, n_features).astype(np.float32)
        raw_flat = scaled_windows.reshape(n_windows, -1)

        # PCA and its post-PCA scaler are fitted ONLY on training recordings.
        pca = PCA(n_components=3, random_state=55)
        pca.fit(raw_flat[train_idx])
        pca_unscaled_all = pca.transform(raw_flat)
        pca_scaler = StandardScaler()
        pca_scaler.fit(pca_unscaled_all[train_idx])
        pca_data = pca_scaler.transform(pca_unscaled_all)

        AE_LATENT_DIM = 8
        encoder, decoder, autoencoder = self._build_trajectory_autoencoder(window_size, n_features, code_size=AE_LATENT_DIM)
        autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

        # Use independent recording-level validation when available. If the
        # dataset contains only one trajectory per group, a leakage-free held-out
        # validation set cannot be formed; in that case training loss is monitored
        # and the limitation is written to the configuration file.
        has_validation = len(val_idx) > 0
        monitor_name = 'val_loss' if has_validation else 'loss'
        early = keras.callbacks.EarlyStopping(
            monitor=monitor_name, patience=15, min_delta=1e-6, restore_best_weights=True
        )
        reduce = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_name, factor=0.5, patience=6, min_lr=1e-6, verbose=1
        )
        fit_kwargs = dict(
            x=scaled_windows[train_idx],
            y=scaled_windows[train_idx],
            epochs=150,
            batch_size=64,
            shuffle=True,
            callbacks=[early, reduce],
            verbose=1,
        )
        if has_validation:
            fit_kwargs['validation_data'] = (scaled_windows[val_idx], scaled_windows[val_idx])
        history = autoencoder.fit(**fit_kwargs)

        # Preserve the reconstruction-pretrained latent codes for an audit trail,
        # then perform unsupervised clustering-aware refinement on TRAINING
        # recordings only. No experimental/group labels are used in refinement.
        encoded_pre_refine_raw = encoder.predict(scaled_windows, batch_size=256, verbose=0)
        pre_refine_scaler = StandardScaler()
        pre_refine_scaler.fit(encoded_pre_refine_raw[train_idx])
        encoded_pre_refine = pre_refine_scaler.transform(encoded_pre_refine_raw)

        # Stage 1: general DEC-style unsupervised refinement.
        refinement_df = self._refine_trajectory_autoencoder_for_clustering(
            encoder, decoder, scaled_windows[train_idx], n_clusters=n_clusters,
            epochs=25, batch_size=256, cluster_weight=0.08, seed=55
        )
        refinement_df.to_csv(
            os.path.join(root_folder, 'trajectory_autoencoder_clustering_refinement_history.csv'),
            index=False
        )

        # Stage 2: GMM-aware unsupervised refinement. This specifically shapes the
        # latent space to be more compatible with Gaussian-mixture clustering,
        # while still using only training recordings and no experimental labels.
        gmm_refinement_df = self._refine_trajectory_autoencoder_for_gmm(
            encoder, decoder, scaled_windows[train_idx], n_clusters=n_clusters,
            epochs=30, batch_size=256, cluster_weight=0.08, compact_weight=0.02, seed=77
        )
        gmm_refinement_df.to_csv(
            os.path.join(root_folder, 'trajectory_autoencoder_GMM_refinement_history.csv'),
            index=False
        )

        # Final clustering-aware latent representation. The latent scaler is fitted
        # on training recordings only and then applied unchanged to all windows.
        encoded_raw = encoder.predict(scaled_windows, batch_size=256, verbose=0)
        latent_scaler = StandardScaler()
        latent_scaler.fit(encoded_raw[train_idx])
        encoded = latent_scaler.transform(encoded_raw)

        partition_labels = np.full(n_windows, 'Training', dtype=object)
        if has_validation:
            partition_labels[val_idx] = 'Validation'
        metadata = metadata.copy()
        metadata['Partition'] = partition_labels

        comparison, predictions, permutation_null = self._evaluate_trajectory_representations(
            raw_flat, pca_data, encoded, true_labels, metadata, n_clusters, n_permutations=1000
        )
        # One final analysis with a direct Before-AE vs After-AE comparison.
        # PCA remains available as an auxiliary representation in Supplementary,
        
        comparison.to_csv(os.path.join(supp_folder, 'full_trajectory_representation_comparison_including_PCA.csv'), index=False)
        main_comparison = comparison[comparison['Representation'].isin(['Raw trajectory', 'Conv1D-AE'])].copy()
        main_comparison.to_csv(os.path.join(main_folder, 'trajectory_before_vs_after_AE_metrics.csv'), index=False)
        main_comparison[['Representation','Method','Evaluation_N_Windows','Evaluation_N_Trajectory_IDs','ARI','NMI','Null_ARI_Mean','Null_ARI_2.5%','Null_ARI_97.5%','ARI_Permutation_P_OneSided']].to_csv(
            os.path.join(main_folder, 'trajectory_external_validation_ARI_NMI_permutation.csv'), index=False
        )

        
        # summarizes the already-generated global cluster assignments and does
        # not refit any model or alter the primary dataset-level scores.
        self._groupwise_internal_metrics_from_global_clustering(
            raw_flat, encoded, metadata, predictions, main_folder
        )

        # Explicit improvement table: positive Delta_ARI/Delta_NMI/Delta_Silhouette
        # and negative Delta_Davies_Bouldin indicate improvement after AE.
        raw_m = main_comparison[main_comparison.Representation == 'Raw trajectory'].set_index('Method')
        ae_m = main_comparison[main_comparison.Representation == 'Conv1D-AE'].set_index('Method')
        common_methods = [m for m in ['KMeans','Birch','GMM','Spectral'] if m in raw_m.index and m in ae_m.index]
        improvement_rows = []
        for method in common_methods:
            improvement_rows.append({
                'Method': method,
                'Raw_ARI': raw_m.loc[method, 'ARI'],
                'AE_ARI': ae_m.loc[method, 'ARI'],
                'Delta_ARI_AE_minus_Raw': ae_m.loc[method, 'ARI'] - raw_m.loc[method, 'ARI'],
                'Raw_NMI': raw_m.loc[method, 'NMI'],
                'AE_NMI': ae_m.loc[method, 'NMI'],
                'Delta_NMI_AE_minus_Raw': ae_m.loc[method, 'NMI'] - raw_m.loc[method, 'NMI'],
                'Raw_Silhouette': raw_m.loc[method, 'Silhouette_within_representation'],
                'AE_Silhouette': ae_m.loc[method, 'Silhouette_within_representation'],
                'Delta_Silhouette_AE_minus_Raw': ae_m.loc[method, 'Silhouette_within_representation'] - raw_m.loc[method, 'Silhouette_within_representation'],
                'Raw_Davies_Bouldin': raw_m.loc[method, 'Davies_Bouldin_within_representation'],
                'AE_Davies_Bouldin': ae_m.loc[method, 'Davies_Bouldin_within_representation'],
                'Delta_Davies_Bouldin_AE_minus_Raw': ae_m.loc[method, 'Davies_Bouldin_within_representation'] - raw_m.loc[method, 'Davies_Bouldin_within_representation'],
            })
        improvement_df = pd.DataFrame(improvement_rows)
        improvement_df['ARI_Improved'] = improvement_df['Delta_ARI_AE_minus_Raw'] > 0
        improvement_df['NMI_Improved'] = improvement_df['Delta_NMI_AE_minus_Raw'] > 0
        improvement_df['Silhouette_Improved'] = improvement_df['Delta_Silhouette_AE_minus_Raw'] > 0
        improvement_df['Davies_Bouldin_Improved'] = improvement_df['Delta_Davies_Bouldin_AE_minus_Raw'] < 0
        improvement_df.to_csv(os.path.join(main_folder, 'raw_vs_AE_improvement_summary.csv'), index=False)
        # Descriptive elbow plots based on the new final trajectory analysis.
        # These figures are not used to choose k in the primary analysis.
        self._save_trajectory_elbow_plots(raw_flat, encoded, supp_folder)
        metadata.to_csv(os.path.join(root_folder, 'trajectory_window_metadata.csv'), index=False)
        partition_table.to_csv(os.path.join(root_folder, 'trajectory_recording_partitions.csv'), index=False)
        latent_columns = [f'Latent_{i+1}' for i in range(AE_LATENT_DIM)]
        pd.DataFrame(encoded, columns=latent_columns).to_csv(
            os.path.join(root_folder, 'trajectory_encoded_data.csv'), index=False
        )
        pd.DataFrame(encoded_pre_refine, columns=latent_columns).to_csv(
            os.path.join(root_folder, 'trajectory_encoded_data_before_clustering_refinement.csv'), index=False
        )
        pd.DataFrame({
            'Component': ['PC1', 'PC2', 'PC3'],
            'Explained_Variance_Ratio': pca.explained_variance_ratio_
        }).to_csv(os.path.join(root_folder, 'trajectory_PCA_explained_variance.csv'), index=False)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(root_folder, 'trajectory_autoencoder_training_history.csv'), index=False)

        # Reconstruction error on independent recording-level partitions.
        train_recon = autoencoder.predict(scaled_windows[train_idx], batch_size=256, verbose=0)
        train_recon_mse = float(np.mean((scaled_windows[train_idx] - train_recon) ** 2))
        reconstruction_rows = [{
            'Partition': 'Training',
            'N_Windows': len(train_idx),
            'N_Trajectory_IDs': metadata.iloc[train_idx].Trajectory_ID.nunique(),
            'Reconstruction_MSE': train_recon_mse
        }]
        if has_validation:
            val_recon = autoencoder.predict(scaled_windows[val_idx], batch_size=256, verbose=0)
            val_recon_mse = float(np.mean((scaled_windows[val_idx] - val_recon) ** 2))
            reconstruction_rows.append({
                'Partition': 'Validation',
                'N_Windows': len(val_idx),
                'N_Trajectory_IDs': metadata.iloc[val_idx].Trajectory_ID.nunique(),
                'Reconstruction_MSE': val_recon_mse
            })
        reconstruction_df = pd.DataFrame(reconstruction_rows)
        reconstruction_df.to_csv(os.path.join(main_folder, 'trajectory_autoencoder_reconstruction_error.csv'), index=False)
        permutation_null.to_csv(os.path.join(supp_folder, 'trajectory_ARI_recording_level_permutation_null.csv'), index=False)

        # 3D cluster plots for the final one-analysis workflow.
        self._save_trajectory_3d_cluster_grid(
            raw_flat, 'Raw trajectory', predictions, metadata,
            os.path.join(main_folder, 'trajectory_clusters_raw_before_AE_3D.png'), train_idx=train_idx, individual_dir=cluster_folder
        )
        self._save_trajectory_3d_cluster_grid(
            encoded, 'Conv1D-AE', predictions, metadata,
            os.path.join(main_folder, 'trajectory_clusters_after_AE_3D.png'), train_idx=train_idx, individual_dir=cluster_folder
        )
        # Keep PCA only as a supplementary visualization.
        self._save_trajectory_3d_cluster_grid(
            pca_data, 'PCA', predictions, metadata,
            os.path.join(supp_folder, 'trajectory_clusters_PCA_3D.png'), train_idx=train_idx
        )

        
        with open(os.path.join(root_folder, 'trajectory_analysis_configuration.txt'), 'w', encoding='utf-8') as f:
            f.write('Analysis unit: fixed-length continuous within-sheet trajectory window\n')
            f.write('Each Excel sheet is treated as one ordered trajectory/recording unit.\n')
            f.write(f'Primary window length: {primary_window} rows/frames\n')
            f.write(f'Primary stride: {max(1, primary_window // 2)} rows/frames (50% overlap)\n')
            f.write('Temporal features: X, Y, Z, speed, acceleration, turning angle, dZ\n')
            f.write('Speed units: coordinate units per frame (no frame-time column assumed)\n')
            f.write('Conv1D operates along the temporal/window dimension.\n')
            f.write('Trajectory AE architecture: GaussianNoise(0.05) -> Conv1D 32/64/128 -> Flatten -> Dense 64 -> latent code.\n')
            f.write(f'Latent dimensions: {AE_LATENT_DIM}\n')
            f.write('AE pretraining: denoising reconstruction MSE, Adam 1e-4, maximum 150 epochs, batch size 64.\n')
            f.write('Clustering-aware refinement: 25-epoch DEC-style refinement followed by 30-epoch unsupervised GMM-aware latent refinement.\n')
            f.write('DEC centers are initialized by K-Means; GMM-aware refinement is initialized by a diagonal-covariance Gaussian mixture fitted to training-recording latent codes only.\n')
            f.write('Experimental group labels, ARI and NMI are NOT used to optimize the autoencoder/refinement.\n')
            f.write('Random seed: 55\n')
            f.write(f'Number of windows: {n_windows}\n')
            f.write(f'Number of trajectory/recording IDs: {metadata.Trajectory_ID.nunique()}\n')
            f.write(f'Training trajectory IDs: {partition_table.loc[partition_table.Partition == "Training", "Trajectory_ID"].nunique()}\n')
            f.write(f'Validation trajectory IDs: {partition_table.loc[partition_table.Partition == "Validation", "Trajectory_ID"].nunique()}\n')
            f.write('Partitioning: whole Trajectory_ID/recording units, stratified within biological group when >=2 recordings are available.\n')
            f.write('StandardScaler, PCA, PCA scaler, and latent scaler are fitted on training recordings only.\n')
            f.write('Conv1D-AE validation uses only held-out recording IDs; no overlapping windows cross partitions.\n')
            if len(val_idx) == 0:
                f.write('WARNING: independent validation was unavailable because the supplied data did not contain enough recording IDs for a held-out split.\n')
            f.write(f'Biological groups ({n_clusters}): {", ".join(groups)}\n')
            f.write('Primary manuscript comparison: Raw trajectory (before AE) versus Conv1D-AE (after AE).\n')
            f.write('PCA is retained only as an auxiliary/supplementary representation.\n')
            f.write('Primary comparison metrics: ARI and NMI; Silhouette/DB remain internal geometric diagnostics.\n')
            f.write('ARI significance: 1000-permutation null with labels shuffled at Trajectory_ID/recording level.\n')
            f.write('Silhouette and Davies-Bouldin are retained only as within-representation geometric diagnostics.\n')
            f.write('Reconstruction error: mean squared error on training and, when available, held-out validation recordings.\n')
            f.write('Important: overlapping windows are analytical units, not independent biological replicates.\n')

        
        
        # reconstruction quality. Internal geometric indices are not used here
        # for cross-representation superiority claims.
        methods_order = ['KMeans', 'Birch', 'GMM', 'Spectral']
        reps_order = ['Raw trajectory', 'Conv1D-AE']
        x = np.arange(len(methods_order)); width = 0.34

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # A: ARI across representations.
        for j, rep in enumerate(reps_order):
            vals = comparison[comparison.Representation == rep].set_index('Method').loc[methods_order, 'ARI'].to_numpy()
            offset = (j - 0.5) * width
            ax1.bar(x + offset, vals, width=width, label=rep, edgecolor='black', linewidth=0.8)
        ax1.set_title('A. External agreement: ARI', fontsize=16, fontweight='bold', pad=12)
        ax1.set_xlabel('Clustering method', fontsize=13, fontweight='bold', labelpad=8)
        ax1.set_ylabel('Adjusted Rand Index (ARI)', fontsize=13, fontweight='bold', labelpad=8)
        ax1.set_xticks(x); ax1.set_xticklabels(methods_order, fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(fontsize=10, frameon=True, loc='upper right'); ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # B: NMI across representations.
        for j, rep in enumerate(reps_order):
            vals = comparison[comparison.Representation == rep].set_index('Method').loc[methods_order, 'NMI'].to_numpy()
            offset = (j - 0.5) * width
            ax2.bar(x + offset, vals, width=width, label=rep, edgecolor='black', linewidth=0.8)
        ax2.set_title('B. External agreement: NMI', fontsize=16, fontweight='bold', pad=12)
        ax2.set_xlabel('Clustering method', fontsize=13, fontweight='bold', labelpad=8)
        ax2.set_ylabel('Normalized Mutual Information (NMI)', fontsize=13, fontweight='bold', labelpad=8)
        ax2.set_xticks(x); ax2.set_xticklabels(methods_order, fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=11)
        ax2.legend(fontsize=10, frameon=True, loc='upper right'); ax2.grid(axis='y', linestyle='--', alpha=0.3)

        # C: recording-level permutation null for the focal Conv1D-AE/KMeans result.
        focal = permutation_null[(permutation_null.Representation == 'Conv1D-AE') &
                                 (permutation_null.Method == 'KMeans')]
        observed_focal = comparison[(comparison.Representation == 'Conv1D-AE') &
                                    (comparison.Method == 'KMeans')].iloc[0]
        if len(focal):
            ax3.hist(focal['Null_ARI'], bins=30, alpha=0.8, edgecolor='black')
            ax3.axvline(observed_focal['ARI'], linestyle='--', linewidth=2.2, label=f"Observed ARI={observed_focal['ARI']:.3f}")
            ax3.axvline(observed_focal['Null_ARI_97.5%'], linestyle=':', linewidth=2.0, label='Null 97.5%')
            ax3.legend(fontsize=10, frameon=True, loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'Permutation null unavailable\n(<2 recording IDs)', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C. Recording-level permutation null', fontsize=16, fontweight='bold', pad=12)
        ax3.set_xlabel('ARI under permuted recording labels', fontsize=12, fontweight='bold', labelpad=8)
        ax3.set_ylabel('Count', fontsize=12, fontweight='bold', labelpad=8)
        ax3.tick_params(axis='both', labelsize=10)

        # D: autoencoder reconstruction error by recording-level partition.
        ax4.bar(reconstruction_df['Partition'], reconstruction_df['Reconstruction_MSE'], edgecolor='black', linewidth=0.8)
        ax4.set_title('D. Clustering-aware Conv1D-AE reconstruction error', fontsize=16, fontweight='bold', pad=12)
        ax4.set_xlabel('Partition', fontsize=12, fontweight='bold', labelpad=8)
        ax4.set_ylabel('Mean squared reconstruction error', fontsize=12, fontweight='bold', labelpad=8)
        ax4.tick_params(axis='both', labelsize=10)
        for tick in ax4.get_xticklabels():
            tick.set_fontweight('bold')
        for i, row in reconstruction_df.reset_index(drop=True).iterrows():
            ax4.text(i, row['Reconstruction_MSE'], f"{row['Reconstruction_MSE']:.4g}", ha='center', va='bottom', fontsize=11, fontweight='bold')

        fig.suptitle('Trajectory-Level Clustering: External Validation and Clustering-Aware Autoencoder', fontsize=20, fontweight='bold', y=0.985)
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.90, wspace=0.30, hspace=0.38)
        fig.savefig(os.path.join(main_folder, 'trajectory_level_main_figure.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)

        
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
        for j, rep in enumerate(reps_order):
            ari_vals = comparison[comparison.Representation == rep].set_index('Method').loc[methods_order, 'ARI'].to_numpy()
            nmi_vals = comparison[comparison.Representation == rep].set_index('Method').loc[methods_order, 'NMI'].to_numpy()
            axes[0].bar(x + (j - 1) * width, ari_vals, width=width, label=rep, edgecolor='black', linewidth=0.8)
            axes[1].bar(x + (j - 1) * width, nmi_vals, width=width, label=rep, edgecolor='black', linewidth=0.8)
        axes[0].set_title('ARI: Raw trajectory vs Conv1D-AE', fontsize=16, fontweight='bold', pad=12); axes[0].set_ylabel('ARI', fontsize=13, fontweight='bold')
        axes[1].set_title('NMI: Raw trajectory vs Conv1D-AE', fontsize=16, fontweight='bold', pad=12); axes[1].set_ylabel('NMI', fontsize=13, fontweight='bold')
        for ax in axes:
            ax.set_xlabel('Clustering method', fontsize=13, fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(methods_order, fontsize=11, fontweight='bold'); ax.tick_params(axis='y', labelsize=11); ax.legend(fontsize=10, frameon=True); ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.90, wspace=0.25)
        fig.savefig(os.path.join(main_folder, 'trajectory_ARI_NMI_main_result.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)

        # ------------------------ Supplementary robustness ----------------------
        sensitivity_rows = []
        for w in [25, 50, 100, 200]:
            sw, smeta, _ = self._load_trajectory_windows(w, max(1, w // 2))
            if len(sw) <= n_clusters or smeta.empty:
                continue
            # Use the same group definition; only groups represented at this window length are retained.
            present = list(pd.unique(smeta['Group']))
            if len(present) < 2:
                continue
            map_w = {g: i for i, g in enumerate(present)}
            sl = smeta['Group'].map(map_w).to_numpy(dtype=int)
            sf = sw.shape[-1]

            # Leakage-safe sensitivity preprocessing: split complete recording IDs
            # first, fit scaling/PCA on training recordings only, and transform
            # validation/all windows with those training-fitted objects.
            s_train_idx, s_val_idx, _ = self._trajectory_train_val_split(
                smeta, val_fraction=0.20, seed=55
            )
            s_scaler = StandardScaler()
            s_scaler.fit(sw[s_train_idx].reshape(-1, sf))
            ss = s_scaler.transform(sw.reshape(-1, sf)).reshape(sw.shape)
            sflat = ss.reshape(len(ss), -1)

            s_pca = PCA(n_components=3, random_state=55)
            s_pca.fit(sflat[s_train_idx])
            spca_unscaled = s_pca.transform(sflat)
            s_pca_scaler = StandardScaler()
            s_pca_scaler.fit(spca_unscaled[s_train_idx])
            spca = s_pca_scaler.transform(spca_unscaled)

            # Sensitivity remains a descriptive clustering robustness analysis on
            # the common transformed dataset; preprocessing itself is train-only.
            cap = min(5000, len(sflat))
            rng = np.random.default_rng(55)
            ids = np.sort(rng.choice(len(sflat), cap, replace=False)) if len(sflat) > cap else np.arange(len(sflat))
            for rep_name, d in [('Raw trajectory', sflat), ('PCA', spca)]:
                pred = KMeans(n_clusters=len(present), n_init=10, random_state=55).fit_predict(d[ids])
                sensitivity_rows.append({
                    'Window_Size_Frames': w,
                    'Stride_Frames': max(1, w // 2),
                    'Representation': rep_name,
                    'N_Windows': len(sw),
                    'N_Trajectory_IDs': smeta.Trajectory_ID.nunique(),
                    'N_Training_Trajectory_IDs': smeta.iloc[s_train_idx].Trajectory_ID.nunique(),
                    'N_Validation_Trajectory_IDs': smeta.iloc[s_val_idx].Trajectory_ID.nunique() if len(s_val_idx) else 0,
                    'N_Groups': len(present),
                    'Silhouette': silhouette_score(d[ids], pred),
                    'Davies_Bouldin': davies_bouldin_score(d[ids], pred),
                    'ARI': adjusted_rand_score(sl[ids], pred),
                    'NMI': normalized_mutual_info_score(sl[ids], pred),
                })

        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(os.path.join(supp_folder, 'trajectory_window_length_sensitivity.csv'), index=False)
        if not sensitivity.empty:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            for rep in ['Raw trajectory', 'PCA']:
                sub = sensitivity[sensitivity.Representation == rep].sort_values('Window_Size_Frames')
                if len(sub):
                    ax.plot(sub.Window_Size_Frames, sub.ARI, marker='o', linewidth=2, label=rep)
            ax.set_title('Supplementary: Trajectory Window-Length Sensitivity')
            ax.set_xlabel('Window length (frames/rows)')
            ax.set_ylabel('Adjusted Rand Index (ARI)')
            ax.set_xticks(sorted(sensitivity.Window_Size_Frames.unique()))
            ax.legend(); ax.grid(True, linestyle='--', alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(supp_folder, 'trajectory_window_length_sensitivity.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close(fig)

        # Additional supplementary metrics table and AE loss plot.
        comparison.to_csv(os.path.join(supp_folder, 'full_trajectory_metrics_all_methods.csv'), index=False)
        comparison[['Representation','Method','Silhouette_within_representation','Davies_Bouldin_within_representation']].to_csv(
            os.path.join(supp_folder, 'internal_geometric_indices_within_representation_only.csv'), index=False
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(history_df['loss'], label='Training loss')
        if 'val_loss' in history_df:
            ax.plot(history_df['val_loss'], label='Validation loss')
        ax.set_title('Supplementary: Conv1D-AE Reconstruction Pretraining History')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Squared Error'); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(supp_folder, 'trajectory_autoencoder_training_history.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)

        if not refinement_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(refinement_df['Epoch'], refinement_df['Total_Loss'], label='Total loss')
            ax.plot(refinement_df['Epoch'], refinement_df['Reconstruction_Loss'], label='Reconstruction')
            ax.plot(refinement_df['Epoch'], refinement_df['Clustering_KL'], label='Clustering KL')
            ax.set_title('Supplementary: Unsupervised Clustering-Aware AE Refinement')
            ax.set_xlabel('Refinement epoch'); ax.set_ylabel('Loss'); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(supp_folder, 'trajectory_autoencoder_clustering_refinement_history.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close(fig)

        print(
            f"[F3DCA] Final trajectory analysis complete: {n_windows} windows, "
            f"window={primary_window}, trajectories={metadata.Trajectory_ID.nunique()}, "
            f"train_trajectories={(partition_table.Partition == 'Training').sum()}, "
            f"validation_trajectories={(partition_table.Partition == 'Validation').sum()}, groups={n_clusters}.",
            flush=True
        )
        return root_folder

    def exit_window(self):
        if hasattr(self, "result_window") and self.result_window.winfo_exists():
            self.result_window.destroy()
        if self.master.winfo_exists():
            self.master.destroy()

    def return_to_main(self):
        if hasattr(self, "result_window") and self.result_window.winfo_exists():
            self.result_window.destroy()
        if self.master.winfo_exists():
            self.master.deiconify()
            self.master.state("zoomed")
            self.master.lift()
    
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()

