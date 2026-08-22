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
        self.ari_running = False
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

        tk.Label(self.control_panel, text="Autoencoder data split:", **label_style).grid(
            row=1, column=0, sticky="w", pady=6, padx=10
        )
        self.entry_subset_autoencoder = tk.Entry(
            self.control_panel, textvariable=self.subset_autoencoder, **entry_style
        )
        self.entry_subset_autoencoder.grid(row=1, column=1, sticky="we", pady=6, padx=10)
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
            self.button_panel, text="Show Results", command=self.show_results, **btn_style
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

    def display_silhouette_kmeans_plot(self):
        self.display_plot("Silhouette kmeans Plot", f"{self.plot_folder}/plot_class_scores_Silhouette_kmeans.png", f"{self.plot_folder}/plot_class_scores_Silhouette_kmeans_EncodedData.png")

    def display_silhouette_birch_plot(self):
        self.display_plot("Silhouette birch Plot", f"{self.plot_folder}/plot_class_scores_Silhouette_birch.png", f"{self.plot_folder}/plot_class_scores_Silhouette_birch_EncodedData.png")

    def display_silhouette_spectral_plot(self):
        self.display_plot("Silhouette spectral Plot", f"{self.plot_folder}/plot_class_scores_Silhouette_spectral.png", f"{self.plot_folder}/plot_class_scores_Silhouette_spectral_EncodedData.png")

    def display_silhouette_gmm_plot(self):
        self.display_plot("Silhouette gmm Plot", f"{self.plot_folder}/plot_class_scores_Silhouette_gmm.png", f"{self.plot_folder}/plot_class_scores_Silhouette_gmm_EncodedData.png")

    def display_davies_bouldin_kmeans_plot(self):
        self.display_plot("Davies Bouldin kmeans Plot", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_kmeans.png", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_kmeans_EncodedData.png")

    def display_davies_bouldin_birch_plot(self):
        self.display_plot("Davies Bouldin birch Plot", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_birch.png", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_birch_EncodedData.png")

    def display_davies_bouldin_spectral_plot(self):
        self.display_plot("Davies Bouldin spectral Plot", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_spectral.png", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_spectral_EncodedData.png")

    def display_davies_bouldin_gmm_plot(self):
        self.display_plot("Davies Bouldin gmm Plot", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_gmm.png", f"{self.plot_folder}/plot_class_scores_Davies Bouldin_gmm_EncodedData.png")

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
            messagebox.showerror("Invalid value", "Autoencoder data split must be greater than 0 and less than or equal to 1.")
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
            raise ValueError("Autoencoder data split must be greater than 0 and less than or equal to 1.")
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

        def ComputeGroupScores(full_data, full_labels, group_names, n_clusters, encoded_bool):
            """Compute within-group clustering scores using merged biological groups."""
            data_array = np.asarray(full_data)
            label_array = np.asarray(full_labels)
            sil_rows, db_rows = [], []
            rng = np.random.default_rng(55)

            for group_id, group_name in enumerate(group_names):
                idx_all = np.flatnonzero(label_array == group_id)
                if len(idx_all) < n_clusters + 1:
                    raise ValueError(f"Not enough observations in group {group_name} for k={n_clusters}.")
                if len(idx_all) > evaluation_cap:
                    idx = np.sort(rng.choice(idx_all, size=evaluation_cap, replace=False))
                else:
                    idx = idx_all
                group_data = data_array[idx]
                print(
                    f"[F3DCA] {'Encoded' if encoded_bool else 'Raw'} group scores: "
                    f"{group_name} ({len(group_data):,} points)...", flush=True
                )

                km_lab = KMeans(n_clusters=n_clusters, n_init=10, random_state=55).fit_predict(group_data)
                bi_lab = Birch(n_clusters=n_clusters, threshold=0.01).fit_predict(group_data)
                gm_lab = GaussianMixture(n_components=n_clusters, n_init=10, random_state=55).fit_predict(group_data)
                sp_lab = SpectralClustering(
                    n_clusters=n_clusters, n_init=10, random_state=55, affinity='rbf'
                ).fit_predict(group_data)

                pred_sets = [km_lab, bi_lab, gm_lab, sp_lab]
                sil_rows.append([silhouette_score(group_data, pred) for pred in pred_sets])
                db_rows.append([davies_bouldin_score(group_data, pred) for pred in pred_sets])

            return np.asarray(sil_rows), np.asarray(db_rows)

        def SaveGroupScores(group_names, silhouette_group, davies_group, data_type):
            methods = ['KMeans', 'Birch', 'GMM', 'Spectral']
            rows = []
            for i, group_name in enumerate(group_names):
                row = {'Group': group_name}
                for j, method in enumerate(methods):
                    row[f'{method}_Silhouette'] = silhouette_group[i, j]
                    row[f'{method}_Davies'] = davies_group[i, j]
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                os.path.join(self.plot_folder, f'group_scores_{data_type}.csv'), index=False
            )

        def PlotClassScores(scores, type, method, encoded_bool, x_labels):
          scores = np.asarray(scores).reshape(-1)
          x_labels = list(x_labels)
          colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'olive', 'teal']

          width = max(10, 1.65 * len(x_labels))
          plt.figure(figsize=(width, 6))
          plt.bar(x_labels, scores, color=colors[0:len(scores)])
          plt.title(type + ' Score for ' + method + ' Clustering')
          plt.xlabel('Biological Group')
          plt.ylabel(type + ' Score')
          plt.xticks(rotation=20, ha='right')
          lower = min(0.0, float(np.min(scores)) - 0.05) if len(scores) else 0
          upper = float(np.max(scores)) + 0.1 if len(scores) else 1
          plt.ylim(lower, upper)
          plt.tight_layout()

          if encoded_bool == False:
              plt.savefig(f"{self.plot_folder}/plot_class_scores_{type}_{method}.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
              plt.close()
          else:
              plt.savefig(f"{self.plot_folder}/plot_class_scores_{type}_{method}_EncodedData.png", dpi=600, bbox_inches='tight', pad_inches=0.35)
              plt.close()

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

        
        raw_group_silhouette, raw_group_davies = ComputeGroupScores(
            analysis_raw, analysis_labels, group_names, n_clusters, encoded_bool=False
        )
        SaveGroupScores(group_names, raw_group_silhouette, raw_group_davies, 'before')

        PlotClassScores(raw_group_silhouette[:,0], 'Silhouette', 'kmeans', False, group_names)
        PlotClassScores(raw_group_silhouette[:,1], 'Silhouette', 'birch', False, group_names)
        PlotClassScores(raw_group_silhouette[:,2], 'Silhouette', 'gmm', False, group_names)
        PlotClassScores(raw_group_silhouette[:,3], 'Silhouette', 'spectral', False, group_names)

        PlotClassScores(raw_group_davies[:,0], 'Davies Bouldin', 'kmeans', False, group_names)
        PlotClassScores(raw_group_davies[:,1], 'Davies Bouldin', 'birch', False, group_names)
        PlotClassScores(raw_group_davies[:,2], 'Davies Bouldin', 'gmm', False, group_names)
        PlotClassScores(raw_group_davies[:,3], 'Davies Bouldin', 'spectral', False, group_names)
        
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
            
        
        encoded_group_silhouette, encoded_group_davies = ComputeGroupScores(
            encoded_data_normalized, analysis_labels, group_names, n_clusters, encoded_bool=True
        )
        SaveGroupScores(group_names, encoded_group_silhouette, encoded_group_davies, 'after')

        PlotClassScores(encoded_group_silhouette[:,0], 'Silhouette', 'kmeans', True, group_names)
        PlotClassScores(encoded_group_silhouette[:,1], 'Silhouette', 'birch', True, group_names)
        PlotClassScores(encoded_group_silhouette[:,2], 'Silhouette', 'gmm', True, group_names)
        PlotClassScores(encoded_group_silhouette[:,3], 'Silhouette', 'spectral', True, group_names)

        PlotClassScores(encoded_group_davies[:,0], 'Davies Bouldin', 'kmeans', True, group_names)
        PlotClassScores(encoded_group_davies[:,1], 'Davies Bouldin', 'birch', True, group_names)
        PlotClassScores(encoded_group_davies[:,2], 'Davies Bouldin', 'gmm', True, group_names)
        PlotClassScores(encoded_group_davies[:,3], 'Davies Bouldin', 'spectral', True, group_names)
        
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

        btn_silhouette_kmeans = tk.Button(self.result_window, text="Silhouette kmeans Plot", command=self.display_silhouette_kmeans_plot)
        btn_silhouette_kmeans.grid(row=1, column=0, pady=10, padx=10)

        btn_silhouette_birch = tk.Button(self.result_window, text="Silhouette birch Plot", command=self.display_silhouette_birch_plot)
        btn_silhouette_birch.grid(row=1, column=1, pady=10, padx=10)

        btn_silhouette_spectral = tk.Button(self.result_window, text="Silhouette spectral Plot", command=self.display_silhouette_spectral_plot)
        btn_silhouette_spectral.grid(row=1, column=2, pady=10, padx=10)

        btn_silhouette_gmm = tk.Button(self.result_window, text="Silhouette gmm Plot", command=self.display_silhouette_gmm_plot)
        btn_silhouette_gmm.grid(row=1, column=3, pady=10, padx=10)

        btn_davies_bouldin_kmeans = tk.Button(self.result_window, text="Davies Bouldin kmeans Plot", command=self.display_davies_bouldin_kmeans_plot)
        btn_davies_bouldin_kmeans.grid(row=2, column=0, pady=10, padx=10)

        btn_davies_bouldin_birch = tk.Button(self.result_window, text="Davies Bouldin birch Plot", command=self.display_davies_bouldin_birch_plot)
        btn_davies_bouldin_birch.grid(row=2, column=1, pady=10, padx=10)

        btn_davies_bouldin_spectral = tk.Button(self.result_window, text="Davies Bouldin spectral Plot", command=self.display_davies_bouldin_spectral_plot)
        btn_davies_bouldin_spectral.grid(row=2, column=2, pady=10, padx=10)

        btn_davies_bouldin_gmm = tk.Button(self.result_window, text="Davies Bouldin gmm Plot", command=self.display_davies_bouldin_gmm_plot)
        btn_davies_bouldin_gmm.grid(row=2, column=3, pady=10, padx=10)

        btn_cross_clusters_silhouette = tk.Button(self.result_window, text="Cross Clusters Silhouette Plot", command=self.display_cross_clusters_silhouette_plot)
        btn_cross_clusters_silhouette.grid(row=3, column=0, pady=10, padx=10)

        btn_cross_clusters_davies = tk.Button(self.result_window, text="Cross Clusters Davies Bouldin Plot", command=self.display_cross_clusters_davies_plot)
        btn_cross_clusters_davies.grid(row=3, column=1, pady=10, padx=10)

        btn_dendrogram_silhouette = tk.Button(self.result_window, text="dendrogram Silhouette", command=self.display_dendrogram_silhouette)
        btn_dendrogram_silhouette.grid(row=4, column=0, pady=10, padx=10)

        btn_dendrogram_davies = tk.Button(self.result_window, text="dendrogram Davies Bouldin", command=self.display_dendrogram_davies)
        btn_dendrogram_davies.grid(row=4, column=1, pady=10, padx=10)
        
        self.ari_pca_button = tk.Button(self.result_window, text="ARI / PCA Analysis", command=self.run_ari_pca_analysis, bg="#1F77B4", fg="white", font=("Helvetica", 11, "bold"))
        self.ari_pca_button.grid(row=4, column=2, columnspan=2, pady=10, padx=10)
        
        
        self.back_button = tk.Button(self.result_window, text="Back to Clustering 3D Data", command=self.return_to_main, bg="black", fg="white")
        self.back_button.grid(row=7, column=3, columnspan=2, pady=10)
        
        
        self.exit_button = tk.Button(self.result_window, text="Exit", command=self.exit_window, bg="red", fg="white")
        self.exit_button.grid(row=8, column=3, columnspan=2, pady=10)
        
        self.result_window.protocol("WM_DELETE_WINDOW", self.return_to_main)
    
    def run_ari_pca_analysis(self):
        if self.ari_running:
            messagebox.showinfo("ARI / PCA Analysis", "ARI / PCA analysis is already running.")
            return
        try:
            self._ari_subset_fraction = float(self.subset_autoencoder.get())
            if not 0 < self._ari_subset_fraction <= 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid value", "Autoencoder data split must be greater than 0 and less than or equal to 1.")
            return
        self.ari_running = True
        self.ari_pca_button.config(state="disabled", text="ARI / PCA Running...")
        threading.Thread(target=self._ari_worker_entry, daemon=True).start()

    def _ari_worker_entry(self):
        try:
            folder = self._run_ari_pca_analysis_worker()
            self.master.after(0, lambda f=folder: self._ari_finished(f))
        except Exception as exc:
            print(traceback.format_exc())
            self.master.after(0, lambda e=str(exc): self._ari_failed(e))

    def _ari_finished(self, ari_folder):
        self.ari_running = False
        if hasattr(self, "ari_pca_button") and self.ari_pca_button.winfo_exists():
            self.ari_pca_button.config(state="normal", text="ARI / PCA Analysis")
        messagebox.showinfo("ARI / PCA Analysis", f"ARI, PCA, and sampling-sensitivity outputs were saved in:\n{ari_folder}")

    def _ari_failed(self, error_message):
        self.ari_running = False
        if hasattr(self, "ari_pca_button") and self.ari_pca_button.winfo_exists():
            self.ari_pca_button.config(state="normal", text="ARI / PCA Analysis")
        messagebox.showerror("ARI / PCA Analysis", error_message)

    def _run_ari_pca_analysis_worker(self):
        try:
            ari_folder = os.path.join(self.plot_folder, "ARI_Comparison")
            os.makedirs(ari_folder, exist_ok=True)

            dfs = []
            for file_name, label in zip(self.file_names, self.file_names):
                xls = pd.ExcelFile(file_name)
                for sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    df = df.iloc[:, :3]
                    df.columns = ['X', 'Y', 'Z']
                    df['label'] = label
                    dfs.append(df)
            concatenated_data = pd.concat(dfs, ignore_index=True)

            subset_fraction = self._ari_subset_fraction
            if not 0 < subset_fraction <= 1:
                raise ValueError("Autoencoder data split must be greater than 0 and less than or equal to 1.")
            subset_data = self.subset_data_consistently(concatenated_data, subset_fraction).dropna()
            xyz = subset_data[['X', 'Y', 'Z']]
            z_scores = np.abs(zscore(xyz))
            keep = (z_scores < 3).all(axis=1)
            xyz = xyz[keep]
            retained_labels = subset_data.loc[keep, 'label']
            raw_same_sample = StandardScaler().fit_transform(xyz)
            true_labels = LabelEncoder().fit_transform(retained_labels)

            encoded_path = os.path.join(self.plot_folder, 'encoded_data.csv')
            if not os.path.exists(encoded_path):
                raise FileNotFoundError("encoded_data.csv was not found. Run the clustering analysis first.")
            encoded_data = pd.read_csv(encoded_path).to_numpy(dtype=float)
            if len(encoded_data) != len(true_labels):
                raise ValueError("Encoded data length does not match the reconstructed labels.")

            pca_data = PCA(n_components=3, random_state=55).fit_transform(raw_same_sample)
            pca_data = StandardScaler().fit_transform(pca_data)
            n_clusters = len(self.file_names)

            def evaluate(data, labels, representation):
                if len(data) > 3000:
                    rng = np.random.default_rng(55)
                    idx = rng.choice(len(data), 3000, replace=False)
                    data_eval = data[idx]
                    labels_eval = labels[idx]
                else:
                    data_eval = data
                    labels_eval = labels
                models = [
                    ('KMeans', KMeans(n_clusters=n_clusters, n_init=10, random_state=55)),
                    ('Birch', Birch(n_clusters=n_clusters, threshold=0.01)),
                    ('GMM', GaussianMixture(n_components=n_clusters, n_init=10, random_state=55)),
                    ('Spectral', SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=55, affinity='rbf'))
                ]
                rows = []
                for name, model in models:
                    pred = model.fit_predict(data_eval)
                    rows.append({
                        'Representation': representation,
                        'Method': name,
                        'Evaluation_N': len(data_eval),
                        'Silhouette': silhouette_score(data_eval, pred),
                        'Davies_Bouldin': davies_bouldin_score(data_eval, pred),
                        'ARI': adjusted_rand_score(labels_eval, pred),
                        'NMI': normalized_mutual_info_score(labels_eval, pred)
                    })
                return pd.DataFrame(rows)

            raw_df = evaluate(raw_same_sample, true_labels, 'Raw')
            pca_df = evaluate(pca_data, true_labels, 'PCA')
            ae_df = evaluate(encoded_data, true_labels, 'Autoencoder')
            comparison = pd.concat([raw_df, pca_df, ae_df], ignore_index=True)
            comparison.to_csv(os.path.join(ari_folder, 'representation_comparison_ARI.csv'), index=False)

            pca_model = PCA(n_components=3, random_state=55).fit(raw_same_sample)
            pd.DataFrame({
                'Component': ['PC1', 'PC2', 'PC3'],
                'Explained_Variance_Ratio': pca_model.explained_variance_ratio_
            }).to_csv(os.path.join(ari_folder, 'PCA_explained_variance.csv'), index=False)

            methods = ['KMeans', 'Birch', 'GMM', 'Spectral']
            x = np.arange(len(methods))
            width = 0.24
            fig, ax = plt.subplots(figsize=(12, 7))
            for j, rep in enumerate(['Raw', 'PCA', 'Autoencoder']):
                vals = comparison[comparison['Representation'] == rep].set_index('Method').loc[methods, 'ARI'].to_numpy()
                ax.bar(x + (j - 1) * width, vals, width=width, label=rep, edgecolor='black', linewidth=1.0)
            ax.set_title('ARI Comparison: Raw vs PCA vs Autoencoder', fontsize=22, fontweight='bold')
            ax.set_xlabel('Clustering Method', fontsize=18, fontweight='bold')
            ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=18, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(prop={'size': 12, 'weight': 'bold'})
            ax.grid(axis='y', linestyle='--', alpha=0.35)
            fig.tight_layout()
            fig.savefig(os.path.join(ari_folder, 'representation_comparison_ARI.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close(fig)

            sensitivity_rows = []
            fractions = [0.05, 0.10, 0.25, 0.50, 1.00]
            for fraction in fractions:
                sampled_parts = []
                for _, group in concatenated_data.groupby('label', sort=False):
                    n = max(1, int(fraction * len(group)))
                    sampled_parts.append(group.sample(n=n, random_state=55))
                sampled = pd.concat(sampled_parts, ignore_index=True).dropna()
                sx = sampled[['X', 'Y', 'Z']]
                z_scores = np.abs(zscore(sx))
                keep = (z_scores < 3).all(axis=1)
                sx = StandardScaler().fit_transform(sx[keep])
                sl = LabelEncoder().fit_transform(sampled.label[keep])
                pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=55).fit_predict(sx)
                if len(sx) > 5000:
                    rng = np.random.default_rng(55)
                    idx = rng.choice(len(sx), 5000, replace=False)
                    sil = silhouette_score(sx[idx], pred[idx])
                else:
                    sil = silhouette_score(sx, pred)
                sensitivity_rows.append({
                    'Sampling_Proportion': fraction,
                    'N_Points': len(sx),
                    'Silhouette': sil,
                    'Davies_Bouldin': davies_bouldin_score(sx, pred),
                    'ARI': adjusted_rand_score(sl, pred)
                })
            sensitivity = pd.DataFrame(sensitivity_rows)
            sensitivity.to_csv(os.path.join(ari_folder, 'sampling_sensitivity.csv'), index=False)

            fig, ax = plt.subplots(figsize=(10, 6.5))
            ax.plot(sensitivity['Sampling_Proportion'] * 100, sensitivity['ARI'], marker='o', linewidth=2.3, markersize=8)
            ax.set_title('ARI Sampling Sensitivity', fontsize=22, fontweight='bold')
            ax.set_xlabel('Sampling Proportion (%)', fontsize=18, fontweight='bold')
            ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=18, fontweight='bold')
            ax.set_xticks([5, 10, 25, 50, 100])
            ax.tick_params(axis='both', labelsize=14)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight('bold')
            ax.grid(True, linestyle='--', alpha=0.35)
            fig.tight_layout()
            fig.savefig(os.path.join(ari_folder, 'sampling_sensitivity.png'), dpi=600, bbox_inches='tight', pad_inches=0.35)
            plt.close(fig)

            return ari_folder
        except Exception:
            raise

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

