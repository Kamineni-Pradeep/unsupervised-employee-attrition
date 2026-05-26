"""
Unsupervised Pattern Study in Employee Attrition
=================================================
Dataset  : IBM HR Analytics Employee Attrition (MySQL database)
Objective: Discover hidden attrition patterns using unsupervised learning
Author   : Pradeep Kamineni
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

# ── MySQL Configuration ────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     'localhost',
    'user':     'root',    # Update with your MySQL username
    'password': '',        # Update with your MySQL password
    'database': 'employee_attrition_db'
}

print("=" * 60)
print("  Unsupervised Pattern Study in Employee Attrition")
print("=" * 60)

os.makedirs('plots', exist_ok=True)

# ── 1. Load Dataset from MySQL ────────────────────────────────────────────────
print("\n[1] Loading Dataset from MySQL...")

df = None
try:
    import mysql.connector
    from sqlalchemy import create_engine
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
    df = pd.read_sql("SELECT * FROM employee_attrition", engine)
    df.drop(columns=['id'], inplace=True, errors='ignore')
    print(f"    Loaded from MySQL: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"    MySQL connection failed: {e}")
    print("    Run db_setup.py first to load data into MySQL")
    print("    Falling back to CSV/generated dataset...")

if df is None:
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/IBM-HR-Analytics-Employee-Attrition-Performance.csv"
    try:
        df = pd.read_csv(url)
        print(f"    Loaded from URL: {df.shape[0]} rows")
    except Exception:
        print("    Generating representative dataset...")
        np.random.seed(42)
        n = 1470
        df = pd.DataFrame({
            'Age': np.random.randint(18, 60, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n),
            'DistanceFromHome': np.random.randint(1, 30, n),
            'Education': np.random.randint(1, 5, n),
            'EnvironmentSatisfaction': np.random.randint(1, 4, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'JobInvolvement': np.random.randint(1, 4, n),
            'JobLevel': np.random.randint(1, 5, n),
            'JobRole': np.random.choice(['Sales Executive','Research Scientist',
                                         'Laboratory Technician','Manager',
                                         'Sales Representative','Human Resources'], n),
            'JobSatisfaction': np.random.randint(1, 4, n),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
            'MonthlyIncome': np.random.randint(1009, 20000, n),
            'NumCompaniesWorked': np.random.randint(0, 9, n),
            'OverTime': np.random.choice(['Yes', 'No'], n),
            'PercentSalaryHike': np.random.randint(11, 25, n),
            'PerformanceRating': np.random.randint(3, 4, n),
            'RelationshipSatisfaction': np.random.randint(1, 4, n),
            'StockOptionLevel': np.random.randint(0, 3, n),
            'TotalWorkingYears': np.random.randint(0, 40, n),
            'TrainingTimesLastYear': np.random.randint(0, 6, n),
            'WorkLifeBalance': np.random.randint(1, 4, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 18, n),
            'YearsSinceLastPromotion': np.random.randint(0, 15, n),
            'YearsWithCurrManager': np.random.randint(0, 17, n),
        })
        print(f"    Generated: {df.shape[0]} rows")

print(f"    Shape       : {df.shape}")
print(f"    Attrition % : {(df['Attrition']=='Yes').mean()*100:.1f}%")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis...")
print(f"    Employees     : {len(df)}")
print(f"    Attrition Yes : {(df['Attrition']=='Yes').sum()}")
print(f"    Attrition No  : {(df['Attrition']=='No').sum()}")
print(f"    Missing Values: {df.isnull().sum().sum()}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Employee Attrition — Exploratory Analysis', fontsize=14, fontweight='bold')

counts = df['Attrition'].value_counts()
axes[0].bar(counts.index, counts.values, color=['#2ecc71','#e74c3c'], edgecolor='white')
axes[0].set_title('Attrition Distribution')
axes[0].set_xlabel('Attrition')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v+10, str(v), ha='center', fontweight='bold')

df[df['Attrition']=='Yes']['Age'].hist(ax=axes[1], bins=20, alpha=0.7, color='#e74c3c', label='Yes')
df[df['Attrition']=='No']['Age'].hist(ax=axes[1], bins=20, alpha=0.7, color='#2ecc71', label='No')
axes[1].set_title('Age Distribution by Attrition')
axes[1].set_xlabel('Age')
axes[1].legend()

dept = df.groupby('Department')['Attrition'].apply(lambda x: (x=='Yes').mean()*100)
axes[2].bar(dept.index, dept.values, color='#3498db', edgecolor='white')
axes[2].set_title('Attrition Rate by Department (%)')
axes[2].set_xlabel('Department')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('plots/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/eda_overview.png")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
print("\n[3] Data Preprocessing...")
drop_cols = ['EmployeeCount','Over18','StandardHours','EmployeeNumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"    Encoded {len(cat_cols)} categorical columns")
print(f"    Scaled  {X_scaled.shape[1]} features across {X_scaled.shape[0]} samples")

# ── 4. PCA ────────────────────────────────────────────────────────────────────
print("\n[4] PCA — Dimensionality Reduction...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
print(f"    PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%  Total={sum(explained):.1f}%")

# ── 5. KMeans ────────────────────────────────────────────────────────────────
print("\n[5] KMeans Clustering...")
inertias, sil_scores = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KMeans — Optimal Cluster Selection', fontsize=14, fontweight='bold')
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(list(k_range), sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/kmeans_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/kmeans_selection.png")

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, cluster_labels)
print(f"    k={optimal_k}  Silhouette={sil:.4f}")
print(f"    Sizes: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")

# ── 6. Visualise Clusters ─────────────────────────────────────────────────────
print("\n[6] Cluster Visualisation...")
colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
names  = ['Cluster 0\n(Stable)','Cluster 1\n(At Risk)','Cluster 2\n(Satisfied)','Cluster 3\n(Transitional)']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('KMeans Cluster Analysis', fontsize=14, fontweight='bold')
for i in range(optimal_k):
    mask = cluster_labels == i
    axes[0].scatter(X_pca[mask,0], X_pca[mask,1], c=colors[i], label=names[i], alpha=0.6, s=30)
axes[0].set_title('PCA — KMeans Clusters')
axes[0].set_xlabel(f'PC1 ({explained[0]:.1f}%)')
axes[0].set_ylabel(f'PC2 ({explained[1]:.1f}%)')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.2)

attrition_c = ['#e74c3c' if a==1 else '#2ecc71' for a in y.values]
axes[1].scatter(X_pca[:,0], X_pca[:,1], c=attrition_c, alpha=0.5, s=30)
axes[1].set_title('PCA — Actual Attrition')
axes[1].set_xlabel(f'PC1 ({explained[0]:.1f}%)')
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor='#e74c3c',label='Yes'), Patch(facecolor='#2ecc71',label='No')])
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig('plots/cluster_visualisation.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/cluster_visualisation.png")

# ── 7. Cluster Profile ────────────────────────────────────────────────────────
print("\n[7] Cluster Profiling...")
df_encoded['Cluster'] = cluster_labels
profile_cols = ['Age','MonthlyIncome','JobSatisfaction','YearsAtCompany',
                'WorkLifeBalance','EnvironmentSatisfaction','Attrition']
profile = df_encoded.groupby('Cluster')[profile_cols].mean().round(2)
print(profile.to_string())

attrition_by_cluster = df_encoded.groupby('Cluster')['Attrition'].mean() * 100
print("\n    Attrition Rate per Cluster (%):")
for c, r in attrition_by_cluster.items():
    print(f"      Cluster {c}: {r:.1f}%")

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(profile.drop(columns=['Attrition']).T, annot=True, fmt='.2f',
            cmap='RdYlGn', ax=ax, linewidths=0.5)
ax.set_title('Cluster Profiles — Feature Means', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/cluster_heatmap.png")

# ── 8. Feature Importance ─────────────────────────────────────────────────────
print("\n[8] Feature Importance...")
corr = df_encoded.drop(columns=['Cluster']).corr()['Attrition'].drop('Attrition')
top = corr.abs().sort_values(ascending=False).head(12)
bar_colors = ['#e74c3c' if corr[f]>0 else '#3498db' for f in top.index]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top.index, corr[top.index], color=bar_colors, edgecolor='white')
ax.set_title('Top 12 Features Correlated with Attrition', fontsize=13, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: plots/feature_importance.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Records        : {len(df)}")
print(f"  Features       : {X_scaled.shape[1]}")
print(f"  Algorithm      : KMeans (k={optimal_k})")
print(f"  Silhouette     : {sil:.4f}")
print(f"  PCA Variance   : {sum(explained):.1f}%")
high = attrition_by_cluster.idxmax()
print(f"  High-Risk      : Cluster {high} ({attrition_by_cluster[high]:.1f}% attrition)")
print("="*60)
print("\nAnalysis complete!")
