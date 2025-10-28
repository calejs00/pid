"""
Script: PCA + KMeans para identificar zonas favorables para instalación de paneles solares.
- Lee un CSV con columna 'mes_latitud_longitud' formateada como 'mes_latitud_longitud'
- Calcula un índice de potencial solar
- Normaliza variables, aplica PCA (2 componentes) y KMeans
- Determina el mejor k por silhouette (opcional)
- Guarda resultados y gráficos

Requisitos:
pip/conda install pandas scikit-learn matplotlib numpy

Ejecutar:
python script_pv_pca_kmeans.py --input datos.csv --k 3

"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler   # 🔹 Cambiado aquí
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='PCA + KMeans para potencial solar')
    parser.add_argument('--input', '-i', required=True, help='CSV de entrada')
    parser.add_argument('--k', '-k', type=int, default=None, help='Número de clusters (si no se indica, se busca automáticamente)')
    parser.add_argument('--max_k', type=int, default=8, help='k máximo para búsqueda automática')
    parser.add_argument('--out_dir', default='out_pv', help='Directorio de salida')
    return parser.parse_args()


def split_id_col(df, idcol='mes_latitud_longitud'):
    # Se espera formato: mes_latitud_longitud (ej: "1_-90.0_2.5")
    parts = df[idcol].str.split('_', expand=True)
    parts.columns = ['mes', 'lat', 'lon']
    parts['mes'] = parts['mes'].astype(float)
    parts['lat'] = parts['lat'].astype(float)
    parts['lon'] = parts['lon'].astype(float)
    df = pd.concat([df, parts], axis=1)
    return df


def compute_solar_potential(df):
    # Fórmula base: pondera radiación penalizando nubosidad y precipitación.
    # Ajusta si tu escala de nube no es 0-100 o la precipitación está en otra unidad.
    # Añadimos un pequeño epsilon para evitar divisiones por cero.
    eps = 1e-9
    rsds = df['rsds_ensemble_mean']
    clt = df['clt_ensemble_mean']  # asumir 0-100
    pr = df['pr_ensemble_mean']

    # Normalizamos pr a una escala manejable si está en m/s o unidades pequeñas
    pr_scaled = pr.copy()
    # Si la mediana es muy pequeña, multiplicamos por 1000 para llevar a mm/día aprox
    if np.median(np.abs(pr_scaled)) < 1e-3:
        pr_scaled = pr_scaled * 1000.0

    # Índice compuesto (puedes ajustar coeficientes según preferencia)
    solar_potential = (rsds * (1 - clt / 100.0 + eps)) / (1 + pr_scaled + eps)

    df['solar_potential'] = solar_potential
    return df


def auto_choose_k(X, k_min=2, k_max=8):
    best_k = None
    best_score = -1
    scores = {}
    for k in range(k_min, min(k_max, len(X)-1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            scores[k] = -1
            continue
        score = silhouette_score(X, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def make_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Scatter geográfico por solar_potential
    plt.figure(figsize=(10,6))
    plt.scatter(df['lon'], df['lat'], c=df['solar_potential'])
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Potencial solar (índice compuesto)')
    cb = plt.colorbar()
    cb.set_label('solar_potential')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'map_solar_potential.png'), dpi=150)
    plt.close()

    # 2) Scatter PCA coloreado por cluster
    plt.figure(figsize=(8,6))
    plt.scatter(df['PC1'], df['PC2'], c=df['cluster'], s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters en espacio PCA')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_clusters.png'), dpi=150)
    plt.close()

    # 3) Boxplots de variables por cluster
    variables = ['rsds_ensemble_mean','clt_ensemble_mean','pr_ensemble_mean','sfcWind_ensemble_mean','tas_ensemble_mean','solar_potential']
    for var in variables:
        plt.figure(figsize=(8,5))
        df.boxplot(column=var, by='cluster')
        plt.title(f'{var} por cluster')
        plt.suptitle('')
        plt.xlabel('Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'box_{var}.png'), dpi=150)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if 'mes_latitud_longitud' not in df.columns:
        raise ValueError("El CSV debe contener la columna 'mes_latitud_longitud'.")

    df = split_id_col(df, 'mes_latitud_longitud')
    df = compute_solar_potential(df)

    # Variables utilizadas para PCA/Clustering
    features = ['rsds_ensemble_mean','clt_ensemble_mean','pr_ensemble_mean','sfcWind_ensemble_mean','tas_ensemble_mean']
    X = df[features].values

    # 🔹 Estandarizar con mismo rango [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # PCA 2 componentes
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PC1'] = X_pca[:,0]
    df['PC2'] = X_pca[:,1]

    # Elegir k
    if args.k is None:
        best_k, scores = auto_choose_k(X_pca, k_min=2, k_max=args.max_k)
        if best_k is None:
            print('No se pudo elegir automáticamente k; usando k=3 por defecto')
            best_k = 3
        print(f'k elegido por silhouette: {best_k}')
        # Guardar scores
        pd.Series(scores).to_csv(os.path.join(args.out_dir, 'silhouette_scores.csv'))
        k = best_k
    else:
        k = args.k

    # KMeans sobre el espacio PCA
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    df['cluster'] = clusters

    # Ordenar clusters por solar_potential promedio (0 = mejor)
    cluster_order = df.groupby('cluster')['solar_potential'].mean().sort_values(ascending=False).index.tolist()
    # Creamos una nueva etiqueta que preserve el ranking: 0 = peor potencial, -1 = mejor
    # Queremos que 0 sea mejor -> invertimos
    cluster_rank_map = {c: rank for rank, c in enumerate(reversed(cluster_order))}
    df['cluster_ranked'] = df['cluster'].map(cluster_rank_map)

    # Guardar resultados
    df.to_csv(os.path.join(args.out_dir, 'results_with_clusters.csv'), index=False)

    make_plots(df, args.out_dir)

    print('Análisis completado.')
    print(f'Resultados guardados en: {args.out_dir}')


if __name__ == '__main__':
    main()
