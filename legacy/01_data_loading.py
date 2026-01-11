"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 1: Încărcarea și Explorarea Datelor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurare vizualizări
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. ÎNCĂRCAREA DATELOR
# ============================================================================

def load_automobile_data():
    """
    Încarcă dataset-ul Automobile din UCI Repository
    """
    # URL către dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    
    # Numele coloanelor conform documentației UCI
    column_names = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
    ]
    
    # Încărcare date
    df = pd.read_csv(url, names=column_names, na_values='?')
    
    print("="*80)
    print("DATASET ÎNCĂRCAT CU SUCCES!")
    print("="*80)
    print(f"\nDimensiuni: {df.shape[0]} înregistrări, {df.shape[1]} atribute")
    
    return df

# ============================================================================
# 2. EXPLORAREA DATELOR (EDA)
# ============================================================================

def explore_data(df):
    """
    Analiză exploratorie a datelor
    """
    print("\n" + "="*80)
    print("EXPLORAREA DATELOR")
    print("="*80)
    
    # Informații generale
    print("\n1. INFORMAȚII GENERALE:")
    print("-" * 80)
    print(df.info())
    
    # Statistici descriptive
    print("\n2. STATISTICI DESCRIPTIVE (VARIABILE NUMERICE):")
    print("-" * 80)
    print(df.describe().round(2))
    
    # Missing values
    print("\n3. VALORI LIPSĂ:")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Percentage': missing_percent[missing > 0]
    }).sort_values('Percentage', ascending=False)
    print(missing_df)
    
    # Distribuția variabilei țintă
    print("\n4. VARIABILA ȚINTĂ (PRICE):")
    print("-" * 80)
    price_stats = df['price'].describe()
    print(price_stats)
    print(f"\nValori lipsă în price: {df['price'].isnull().sum()}")
    
    # Variabile categoriale
    print("\n5. VARIABILE CATEGORIALE:")
    print("-" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique()} valori unice")
        print(df[col].value_counts().head())
    
    return missing_df

# ============================================================================
# 3. VIZUALIZĂRI
# ============================================================================

def create_visualizations(df):
    """
    Creează vizualizări pentru analiza datelor
    """
    print("\n" + "="*80)
    print("GENERARE VIZUALIZĂRI")
    print("="*80)
    
    # Eliminăm înregistrările cu price lipsă pentru vizualizări
    df_viz = df.dropna(subset=['price'])
    
    # Figure 1: Distribuția prețului
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df_viz['price'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Preț ($)', fontsize=12)
    axes[0].set_ylabel('Frecvență', fontsize=12)
    axes[0].set_title('Distribuția Prețurilor Automobilelor', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df_viz['price'], vert=True)
    axes[1].set_ylabel('Preț ($)', fontsize=12)
    axes[1].set_title('Box Plot - Identificarea Outliers', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: price_distribution.png")
    plt.show()
    
    # Figure 2: Top 10 mărci după preț mediu
    fig, ax = plt.subplots(figsize=(12, 6))
    make_price = df_viz.groupby('make')['price'].mean().sort_values(ascending=False).head(10)
    make_price.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Marca', fontsize=12)
    ax.set_ylabel('Preț Mediu ($)', fontsize=12)
    ax.set_title('Top 10 Mărci după Preț Mediu', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('top_brands_price.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: top_brands_price.png")
    plt.show()
    
    # Figure 3: Matrice de corelație
    numeric_cols = df_viz.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_viz[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Matricea de Corelație - Variabile Numerice', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: correlation_matrix.png")
    plt.show()
    
    # Figure 4: Relații importante cu prețul
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Engine size vs Price
    axes[0, 0].scatter(df_viz['engine-size'], df_viz['price'], alpha=0.6, edgecolor='black')
    axes[0, 0].set_xlabel('Engine Size', fontsize=11)
    axes[0, 0].set_ylabel('Price ($)', fontsize=11)
    axes[0, 0].set_title('Engine Size vs Price', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Horsepower vs Price
    df_hp = df_viz.dropna(subset=['horsepower'])
    axes[0, 1].scatter(df_hp['horsepower'], df_hp['price'], alpha=0.6, 
                       edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Horsepower', fontsize=11)
    axes[0, 1].set_ylabel('Price ($)', fontsize=11)
    axes[0, 1].set_title('Horsepower vs Price', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Curb weight vs Price
    axes[1, 0].scatter(df_viz['curb-weight'], df_viz['price'], alpha=0.6, 
                       edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Curb Weight', fontsize=11)
    axes[1, 0].set_ylabel('Price ($)', fontsize=11)
    axes[1, 0].set_title('Curb Weight vs Price', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Body style vs Price
    body_price = df_viz.groupby('body-style')['price'].mean().sort_values()
    body_price.plot(kind='barh', ax=axes[1, 1], color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Preț Mediu ($)', fontsize=11)
    axes[1, 1].set_ylabel('Body Style', fontsize=11)
    axes[1, 1].set_title('Body Style vs Preț Mediu', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('price_relationships.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: price_relationships.png")
    plt.show()

# ============================================================================
# 4. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală de execuție
    """
    print("\n" + "="*80)
    print(" " * 20 + "PROIECT MACHINE LEARNING")
    print(" " * 15 + "Predicția Prețului Automobilelor")
    print("="*80)
    
    # Încărcare date
    df = load_automobile_data()
    
    # Explorare
    missing_info = explore_data(df)
    
    # Vizualizări
    create_visualizations(df)
    
    print("\n" + "="*80)
    print("ETAPA 1 FINALIZATĂ CU SUCCES!")
    print("="*80)
    print("\nFișiere generate:")
    print("  • price_distribution.png")
    print("  • top_brands_price.png")
    print("  • correlation_matrix.png")
    print("  • price_relationships.png")
    print("\nUrmătorul pas: Preprocessing și curățarea datelor")
    print("="*80)
    
    return df

# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    df_automobile = main()