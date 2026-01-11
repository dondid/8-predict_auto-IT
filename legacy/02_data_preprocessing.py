"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 2: Preprocessing și Pregătirea Datelor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# ============================================================================
# 1. CURĂȚAREA DATELOR
# ============================================================================

def clean_data(df):
    """
    Curăță datele: tratează missing values și outliers
    """
    print("\n" + "="*80)
    print("CURĂȚAREA DATELOR")
    print("="*80)
    
    # Copie pentru a nu modifica originalul
    df_clean = df.copy()
    
    # 1. Eliminăm înregistrările cu price lipsă (variabila țintă)
    print(f"\n1. Eliminare înregistrări cu price lipsă: {df_clean['price'].isnull().sum()}")
    df_clean = df_clean.dropna(subset=['price'])
    print(f"   Dimensiune după eliminare: {df_clean.shape}")
    
    # 2. Tratarea missing values pentru variabile numerice
    print("\n2. Tratarea missing values pentru variabile numerice:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            # Imputare cu mediana
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            print(f"   • {col}: {missing_count} valori lipsă → imputate cu mediana ({median_value:.2f})")
    
    # 3. Tratarea missing values pentru variabile categoriale
    print("\n3. Tratarea missing values pentru variabile categoriale:")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            # Imputare cu modul
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"   • {col}: {missing_count} valori lipsă → imputate cu modul ({mode_value})")
    
    # 4. Verificare finală
    print("\n4. Verificare finală:")
    print(f"   Total missing values: {df_clean.isnull().sum().sum()}")
    print(f"   Dimensiune finală: {df_clean.shape}")
    
    return df_clean

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """
    Creează features noi relevante
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_fe = df.copy()
    
    # 1. Power-to-Weight Ratio (indicator important pentru performanță)
    df_fe['power-to-weight'] = df_fe['horsepower'] / df_fe['curb-weight']
    print("\n1. Creat: power-to-weight ratio")
    
    # 2. Engine Efficiency (km per liter combinat)
    df_fe['fuel-efficiency'] = (df_fe['city-mpg'] + df_fe['highway-mpg']) / 2
    print("2. Creat: fuel-efficiency (average mpg)")
    
    # 3. Size Category (bazat pe lungime)
    df_fe['size-category'] = pd.cut(df_fe['length'], 
                                     bins=[0, 170, 185, 200, 250],
                                     labels=['compact', 'mid-size', 'large', 'luxury'])
    print("3. Creat: size-category")
    
    # 4. Price Category pentru analiză (nu va fi folosită în training)
    df_fe['price-category'] = pd.cut(df_fe['price'],
                                      bins=[0, 10000, 20000, 50000],
                                      labels=['budget', 'mid-range', 'premium'])
    
    # 5. Luxury Brand Indicator
    luxury_brands = ['bmw', 'mercedes-benz', 'jaguar', 'porsche', 'audi']
    df_fe['is-luxury'] = df_fe['make'].str.lower().isin(luxury_brands).astype(int)
    print("4. Creat: is-luxury (indicator marcă premium)")
    
    print(f"\nTotal features după engineering: {df_fe.shape[1]}")
    
    return df_fe

# ============================================================================
# 3. ENCODING VARIABILE CATEGORIALE
# ============================================================================

def encode_categorical(df, target_col='price'):
    """
    Encodează variabilele categoriale
    """
    print("\n" + "="*80)
    print("ENCODING VARIABILE CATEGORIALE")
    print("="*80)
    
    df_encoded = df.copy()
    
    # Separăm target-ul și price-category
    y = df_encoded[target_col]
    df_encoded = df_encoded.drop([target_col, 'price-category'], axis=1)
    
    # Identificăm variabilele categoriale
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    print(f"\nVariabile categoriale de encodat: {len(categorical_cols)}")
    
    # One-Hot Encoding pentru variabile cu puține categorii
    # Label Encoding pentru variabile cu multe categorii
    
    label_encode_cols = ['make', 'engine-type', 'fuel-system']
    onehot_encode_cols = [col for col in categorical_cols if col not in label_encode_cols]
    
    # Label Encoding
    label_encoders = {}
    for col in label_encode_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            print(f"  • Label Encoding: {col} → {col}_encoded")
            df_encoded = df_encoded.drop(col, axis=1)
    
    # One-Hot Encoding
    print("\n  One-Hot Encoding pentru:")
    for col in onehot_encode_cols:
        if col in df_encoded.columns:
            print(f"  • {col}")
    
    df_encoded = pd.get_dummies(df_encoded, columns=onehot_encode_cols, drop_first=True)
    
    print(f"\nTotal features după encoding: {df_encoded.shape[1]}")
    
    return df_encoded, y, label_encoders

# ============================================================================
# 4. SCALING
# ============================================================================

def scale_features(X_train, X_val, X_test):
    """
    Scalează features cu StandardScaler
    """
    print("\n" + "="*80)
    print("SCALING FEATURES")
    print("="*80)
    
    # Inițializare scaler
    scaler = StandardScaler()
    
    # Fit pe training, transform pe toate
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scalate cu StandardScaler")
    print(f"  Mean: ~0, Std: ~1")
    
    # Convertim înapoi la DataFrame pentru a păstra numele coloanelor
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================================================
# 5. SPLIT DATASET
# ============================================================================

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Împarte datele în train, validation și test
    """
    print("\n" + "="*80)
    print("ÎMPĂRȚIREA DATELOR")
    print("="*80)
    
    # Calculăm proporția pentru validation din train+val
    val_size_adjusted = val_size / (1 - test_size)
    
    # Prima împărțire: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # A doua împărțire: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"\nDimensiuni:")
    print(f"  • Training:   {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  • Validation: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  • Test:       {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"  • Features:   {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# 6. PIPELINE COMPLET
# ============================================================================

def preprocessing_pipeline(df):
    """
    Pipeline complet de preprocessing
    """
    print("\n" + "="*80)
    print(" " * 25 + "PIPELINE PREPROCESSING")
    print("="*80)
    
    # 1. Curățare
    df_clean = clean_data(df)
    
    # 2. Feature Engineering
    df_fe = feature_engineering(df_clean)
    
    # 3. Encoding
    X, y, encoders = encode_categorical(df_fe)
    
    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 5. Scaling
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Salvare obiecte pentru refolosire
    print("\n" + "="*80)
    print("SALVARE OBIECTE")
    print("="*80)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Salvat: scaler.pkl")
    
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("✓ Salvat: encoders.pkl")
    
    # Salvare date procesate
    data_dict = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print("✓ Salvat: processed_data.pkl")
    
    print("\n" + "="*80)
    print("PREPROCESSING FINALIZAT CU SUCCES!")
    print("="*80)
    
    return data_dict

# ============================================================================
# 7. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală
    """
    # Încărcare date
    print("Încărcare date...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    column_names = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
    ]
    df = pd.read_csv(url, names=column_names, na_values='?')
    
    # Execută pipeline
    data_dict = preprocessing_pipeline(df)
    
    print("\nRezumat final:")
    print(f"  Training samples:   {len(data_dict['X_train'])}")
    print(f"  Validation samples: {len(data_dict['X_val'])}")
    print(f"  Test samples:       {len(data_dict['X_test'])}")
    print(f"  Total features:     {data_dict['X_train'].shape[1]}")
    
    return data_dict

# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    processed_data = main()