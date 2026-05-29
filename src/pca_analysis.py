from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca(X, output_dir='reports'):
    Path(output_dir).mkdir(exist_ok=True)
    pca = PCA()
    pca.fit(X)
    df = pd.DataFrame({
        'component': range(1, len(pca.explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': pca.explained_variance_ratio_.cumsum()
    })
    df.to_csv(f'{output_dir}/pca_variance.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df['component'], df['cumulative_variance'], marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_cumulative_variance.png', dpi=150)
    plt.close()
    return df
