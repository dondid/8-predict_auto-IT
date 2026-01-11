import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

# Page Configuration
st.set_page_config(
    page_title="Auto Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data():
    loader = DataLoader()
    return loader.load_data()

def load_metrics():
    try:
        return pd.read_csv(REPORTS_DIR / "final_metrics.csv")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model(model_name):
    model_path = MODELS_DIR / f"{model_name.lower()}_model.pkl"
    return joblib.load(model_path)

def predict_price(model_name, input_data):
    try:
        # Load Model
        model = load_model(model_name)
        
        # Preprocess Input
        preprocessor = DataPreprocessor()
        processed_input = preprocessor.transform_new_data(input_data)
        
        # Predict
        prediction = model.predict(processed_input)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Sidebar
st.sidebar.title("🚗 Auto Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigare", ["Data Explorer", "Detailed Statistics", "Brand Encyclopedia", "Model Performance", "Financial Analysis", "Live Prediction", "🤖 AI Assistant"])
st.sidebar.markdown("---")
st.sidebar.info("Proiect Machine Learning\nPredicția Prețului Automobilelor")

# Export Section
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Kit Prezentare")

def create_export_zip():
    import zipfile
    import io
    from src.config import REPORTS_DIR
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Add Guide
        if os.path.exists("GHID_PREZENTARE.md"):
            zip_file.write("GHID_PREZENTARE.md", "GHID_PREZENTARE.md")
            
        # 2. Add Metrics
        metrics_path = REPORTS_DIR / "final_metrics.csv"
        if metrics_path.exists():
            zip_file.write(metrics_path, "evaluare_performanta.csv")
            
        # 3. Add Modern Comparison (Generated on fly)
        try:
            from src.data.loader import DataLoader
            modern_df = DataLoader().load_modern_data()
            if modern_df is not None:
                modern_avg = modern_df.groupby('Make')['Price'].mean().reset_index()
                modern_avg.columns = ['Brand', 'AvgPrice_2024']
                
                # Get old data
                old_df = load_data()
                old_avg = old_df.groupby('make')['price'].mean().reset_index()
                old_avg.columns = ['Brand', 'AvgPrice_1985']
                old_avg['Brand'] = old_avg['Brand'].str.capitalize()
                
                # Merge
                comp_df = pd.merge(old_avg, modern_avg, on='Brand', how='inner')
                comp_csv = comp_df.to_csv(index=False)
                zip_file.writestr("market_evolution.csv", comp_csv)
        except Exception as e:
            print(f"Export error: {e}")
            
    buffer.seek(0)
    return buffer

zip_buffer = create_export_zip()
st.sidebar.download_button(
    label="⬇️ Descarcă Resurse",
    data=zip_buffer,
    file_name="auto_predictor_presentation_kit.zip",
    mime="application/zip",
    help="Descarcă tabele, grafice și ghidul de prezentare."
)

# Main Content
if page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Explorează setul de date UCI Automobile pentru a înțelege distribuția și relațiile dintre variabile.")
    
    df = load_data()
    
    # --- INTERACTIVE FILTERS ---
    st.sidebar.markdown("### 🔎 Filtrează Datele")
    use_filters = st.sidebar.checkbox("Activează Filtrare Avansată", False)
    
    if use_filters:
        # 1. Brand Filter
        brands = sorted(df['make'].unique())
        sel_brands = st.sidebar.multiselect("Alege Mărci", brands)
        if sel_brands:
            df = df[df['make'].isin(sel_brands)]
            
        # 2. Price Filter
        min_p, max_p = int(df['price'].min()), int(df['price'].max())
        price_range = st.sidebar.slider("Interval Preț ($)", min_p, max_p, (min_p, max_p))
        df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
        
        # 3. Fuel Filter
        fuels = df['fuel-type'].unique()
        sel_fuel = st.sidebar.multiselect("Combustibil", fuels)
        if sel_fuel:
            df = df[df['fuel-type'].isin(sel_fuel)]
            
        st.sidebar.caption(f"Rezultat: {len(df)} vehicule.")

    # metrics row (Updated to use filtered df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cars", len(df))
    c2.metric("Features", df.shape[1])
    # Handle empty df case
    if not df.empty:
        c3.metric("Average Price", f"${df['price'].mean():,.0f}")
        c4.metric("Brands", df['make'].nunique())
    else:
        c3.metric("Average Price", "$0")
        c4.metric("Brands", "0")
    
    st.markdown("---")
    
    # Plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuția Prețurilor")
        fig = px.histogram(df, x="price", nbins=30, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, width='stretch')
        
    with col2:
        st.subheader("Top 10 Mărci (Preț Mediu)")
        top_brands = df.groupby('make')['price'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_brands, x='make', y='price', color='price', color_continuous_scale='Reds')
        st.plotly_chart(fig, width='stretch')
        
    # --- SAFETY ANALYSIS SECTION (New) ---
    st.markdown("### 🛡️ Analiză Siguranță & Risc")
    # Quick calc for plotting
    df['safety_index'] = (df['curb-weight'] / df['curb-weight'].max() * 6) + (1 - (df['normalized-losses']/256) * 4)
    
    col_safe1, col_safe2 = st.columns(2)
    with col_safe1:
        st.caption("Top 5 Cele mai sigure branduri (Estimare)")
        safe_brands = df.groupby('make')['safety_index'].mean().sort_values(ascending=False).head(5)
        st.bar_chart(safe_brands)
        
    with col_safe2:
        st.caption("Preț vs Risc (Normalized Losses)")
        fig_risk = px.scatter(df, x="normalized-losses", y="price", color="make", 
                              title="Riscuri ridicate (Dreapta) vs Preț", hover_data=['make'])
        st.plotly_chart(fig_risk, use_container_width=True)
        
    st.info("ℹ️ **Safety Index**: Calculat pe baza greutății și istoricului de daune (Normalized Losses).")
    st.markdown("---")
        
    # Interactive Scatter
    st.subheader("Relații între Variabile")
    x_axis = st.selectbox("Alege variabila X", ['horsepower', 'engine-size', 'city-mpg', 'curb-weight'], index=0)
    y_axis = st.selectbox("Alege variabila Y", ['price', 'horsepower'], index=0)
    
    # Drop rows with NaN in the selected columns or price (used for size)
    df_plot = df.dropna(subset=[x_axis, y_axis, 'price', 'body-style'])
    
    fig = px.scatter(df_plot, x=x_axis, y=y_axis, color="body-style", size="price", hover_data=['make'])
    st.plotly_chart(fig, use_container_width=True)
    
    # --- UNSUPERVISED LEARNING SECTION (BONUS) ---
    st.markdown("---")
    st.subheader("🧩 Unsupervised Learning: Segmentare Piață (K-Means Clustering)")
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Prepare Data for Clustering
        cluster_feats = ['price', 'horsepower', 'curb-weight', 'highway-mpg']
        # Filter numeric only just in case, though these are numeric
        X_cluster = df[cluster_feats].copy()
        
        # Handle Missing Values
        imputer = SimpleImputer(strategy='mean')
        X_cluster_imp = imputer.fit_transform(X_cluster)
        
        # Scale Data (Crucial for K-Means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster_imp)
        
        # Interactive K Selection
        col_k1, col_k2 = st.columns([1, 3])
        with col_k1:
            k = st.slider("Număr de Clustere (K)", 2, 6, 4, help="În câte segmente să împartă AI-ul piața??")
            
        with col_k2:
            # Fit Model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            X_cluster['Cluster'] = clusters
            # Sort cluster labels by mean price so Cluster 0 is cheapest, etc. (Visual consistency)
            # This is a bit complex to map back efficiently in one line, skipping for robustness.
            X_cluster['Cluster_Label'] = "Segment " + X_cluster['Cluster'].astype(str)
            X_cluster['Make'] = df['make']
            X_cluster['city-mpg'] = df['city-mpg']
            
            st.caption(f"Algoritmul **K-Means** a grupat automat vehiculele în **{k} segmente** bazate pe preț, putere și greutate.")
            
        # Visualization
        fig_clus = px.scatter(X_cluster, x='horsepower', y='price', color='Cluster_Label',
                              size='curb-weight', hover_data=['Make', 'city-mpg'],
                              title=f"Segmentare Automată a Pieței (K={k})",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                              
        st.plotly_chart(fig_clus, use_container_width=True)
        
        # Cluster Interpretation
        st.info(f"💡 Interpretare: K-Means a descoperit natural categoriile (ex: Mașini Economice, Sport, Lux) fără a i se spune ce sunt.")
        
    except Exception as e:
        st.error(f"Eroare la Clustering: {e}")

elif page == "Detailed Statistics":
    import statsmodels.api as sm
    
    st.title("📈 Analiză Statistică Avansată")
    st.markdown("Analiză de regresie OLS (Ordinary Least Squares) pentru a identifica impactul exact al fiecărei variabile asupra prețului.")
    
    # Load and Clean data to avoid losing rows due to NaNs
    raw_df = load_data()
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(raw_df) # Impute missing values
    
    # Select Numerical Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numeric_cols: numeric_cols.remove('price')
    
    st.sidebar.subheader("Configurare Regresie")
    selected_features = st.sidebar.multiselect("Selectează Predictorii (X)", numeric_cols, default=['horsepower', 'engine-size', 'city-mpg', 'curb-weight'])
    
    if len(selected_features) > 0:
        X = df[selected_features]
        y = df['price']
        
        # Add constant for OLS
        X = sm.add_constant(X)
        
        # Fit Model
        model = sm.OLS(y, X).fit()
        
        # Display Summary
        st.subheader("Rezultate Regresie (OLS)")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared", f"{model.rsquared:.4f}")
        col2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
        col3.metric("F-statistic", f"{model.fvalue:.2f}")
        
        st.text(model.summary())
        
        st.markdown("""
        > [!TIP]
        > **P>|t|**: Probabilitatea ca variabila să NU aibă impact. Dacă e < 0.05, variabila e semnificativă statistic.
        > **Coef**: Cu cât crește prețul dacă variabila crește cu o unitate.
        """)
        
        # Residual Plots
        st.subheader("Analiza Reziduurilor")
        residuals = model.resid
        fig = px.scatter(x=model.fittedvalues, y=residuals, labels={'x': 'Predicted Price', 'y': 'Residuals'}, title="Residuals vs Fitted")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Te rog selectează cel puțin o variabilă predictor.")

elif page == "Brand Encyclopedia":
    import google.generativeai as genai
    from src.config import GEMINI_API_KEY
    
    st.title("📚 Brand Encyclopedia & History")
    st.markdown("Analiză combinată: Date Istorice (1985) + Cunoștințe AI despre Evoluție, Fiabilitate și Siguranță.")

    df = load_data()
    top_brands = df['make'].value_counts().index.tolist()
    
    selected_brand = st.selectbox("Alege un Brand pentru Analiză Completă", top_brands)
    
    # 1. Internal Statistics (1985)
    brand_data = df[df['make'] == selected_brand]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Modele în 1985", len(brand_data))
    col1.metric("Preț Mediu '85", f"${brand_data['price'].mean():,.0f}")
    
    col2.metric("Putere Medie (HP)", f"{brand_data['horsepower'].mean():.0f} cp")
    col2.metric("Consum Mediu", f"{brand_data['city-mpg'].mean():.1f} mpg")
    
    # Safety Proxy
    risk_score = brand_data['normalized-losses'].mean()
    col3.metric("Safety/Risk Index", f"{risk_score:.0f}", delta="Scor Mic = Mai Sigur" if risk_score < 120 else "Risc Ridicat", delta_color="inverse")
    col3.metric("Greutate Medie", f"{brand_data['curb-weight'].mean():.0f} lbs")
    
    # DOWNLOAD BUTTON FOR RAW DATA
    csv = brand_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descarcă Datele Brute pt. acest Brand (CSV)",
        csv,
        f"{selected_brand}_1985_data.csv",
        "text/csv",
        key='download-brand'
    )
    
    st.markdown("---")
    
    st.markdown("---")
    
    # 2. AI Generated Intelligence Report (Structured)
    st.markdown("---")
    st.subheader(f"🧠 {selected_brand.upper()}: Raport Profesional & Evoluție")
    
    # Modern Data Context
    from src.data.loader import DataLoader
    modern_df = DataLoader().load_modern_data()
    modern_stats = None
    
    if modern_df is not None:
        # Filter for brand (case insensitive)
        brand_modern = modern_df[modern_df['make_norm'] == selected_brand.lower()]
        if not brand_modern.empty:
            avg_price_2024 = brand_modern['Price'].mean()
            models_list = ", ".join(brand_modern['Model'].unique()[:5])
            modern_stats = {
                'price': avg_price_2024,
                'models': models_list
            }
            # Display Comparison Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Preț Mediu 1985", f"${brand_data['price'].mean():,.0f}")
            with col_m2:
                st.metric("Preț Mediu 2024 (Est.)", f"${avg_price_2024:,.0f}", delta=f"{((avg_price_2024/brand_data['price'].mean())-1)*100:.0f}%")
            with col_m3:
                st.metric("Modele Moderne", f"{len(brand_modern)} găsite")
            
            st.caption(f"Modele recente analizate: {models_list}...")
        else:
            st.info(f"Nu există date recente (2020-2024) pentru {selected_brand} in dataset-ul modern.")
    
    # Live Market Data
    from src.data.live_api import LiveMarket
    market = LiveMarket()
    stock_data = market.get_stock_data(selected_brand)
    live_context_str = ""
    
    if stock_data:
        st.markdown("---")
        st.subheader(f"🌐 Live Market: {selected_brand.upper()} (Financial)")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            delta_color = "normal"
            if stock_data['change_p'] > 0: delta_color = "inverse" # Streamlit metric delta color opposite? No, normal green for up.
            
            st.metric(
                label=f"Acțiuni ({stock_data['symbol']})", 
                value=f"{stock_data['price']:.2f} {stock_data['currency']}",
                delta=f"{stock_data['change_p']:.2f}%"
            )
        
        with col_s2:
            if stock_data['news']:
                with st.expander("📰 Știri Financiare Recente (Live)"):
                    for n in stock_data['news']:
                        st.markdown(f"- [{n['title']}]({n['link']}) ({n['publisher']})")
        
        live_context_str = f"Stock: {stock_data['symbol']} @ {stock_data['price']} {stock_data['currency']} ({stock_data['change_p']}%). Știri recente: {[n['title'] for n in stock_data['news']]}."

    from src.ai.gemini_service import GeminiService
    ai_service = GeminiService()
    
    if ai_service.is_configured:
        stats = {
            'price': brand_data['price'].mean(),
            'hp': brand_data['horsepower'].max()
        }
        
        # Combine contexts
        full_modern_stats = modern_stats.copy() if modern_stats else {}
        if live_context_str:
            full_modern_stats['live_market'] = live_context_str
        
        with st.spinner(f"AI Analizează evoluția și datele LIVE pentru {selected_brand}..."):
            report = ai_service.generate_brand_report(selected_brand, stats, full_modern_stats)
            text_part, evolution_df = ai_service.parse_evolution_csv(report)
            
            # Render Text
            st.markdown(text_part)
            
            # Render Chart
            if evolution_df is not None:
                st.subheader("📈 Proiecție Evoluție Valoare (AI Estimation)")
                st.line_chart(evolution_df.set_index("An"))
                
                # Download
                proj_csv = evolution_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                     "📥 Descarcă Proiecția Evoluției (CSV)",
                     proj_csv,
                     f"{selected_brand}_value_evolution.csv",
                     "text/csv",
                     key='download-evo'
                )
    else:
        st.warning("Activează Online Mode pentru a primi rapoartele de evoluție și graficele de depreciere generate de AI.")


elif page == "Model Performance":
    st.title("🤖 Model Performance")
    
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        # Best Model
        best_model = metrics_df.loc[metrics_df['R2'].idxmax()]
        st.success(f"🏆 Cel mai bun model: **{best_model['Model']}** cu R2 = {best_model['R2']:.4f}")
        
        # Display Table
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R2'], color='lightgreen')
                     .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen'))
        
        # Comparision Chart
        st.subheader("Comparare R2 Score")
        fig = px.bar(metrics_df, x='Model', y='R2', color='Model', range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Comparare RMSE (Eroare)")
        fig = px.bar(metrics_df, x='Model', y='RMSE', color='Model')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Nu există rezultate salvate. Rulează pipeline-ul de antrenare mai întâi.")
        if st.button("Rulează Pipeline Acum"):
            with st.spinner("Antrenare în curs... poate dura 1 minut"):
                import main
                main.main() # This assumes main works as a module call
                st.success("Gata! Reîncarcă pagina.")
    
    # Statistical Compliance Section
    st.markdown("---")
    st.subheader("📚 Analiză Statistică Comparativă (Requirement Compliance)")
    
    wilcoxon_path = REPORTS_DIR / "wilcoxon_results.csv"
    if wilcoxon_path.exists():
        st.write("Rezultatele testului **Wilcoxon Signed-Rank** (pentru a demonstra diferențele semnificative între modele):")
        w_df = pd.read_csv(wilcoxon_path)
        
        # Color p-values
        def highlight_significant(val):
            color = 'lightgreen' if val < 0.05 else 'white'
            return f'background-color: {color}'
        
        st.dataframe(w_df.style.applymap(highlight_significant, subset=['p-value']))
        st.info("💡 Notă: Un p-value < 0.05 indică faptul că diferența de performanță între cele două modele este statistic semnificativă.")
    else:
        st.info("Rulează pipeline-ul cu `python main.py --compare` pentru a genera testele statistice.")

elif page == "Live Prediction":
    st.title("🔮 Live Prediction")
    st.markdown("Configurează mașina și află prețul estimat folosind modelul XGBoost.")
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            make = st.selectbox("Marca", ['bmw', 'audi', 'toyota', 'volvo', 'mercedes-benz', 'nissan'])
            body_style = st.selectbox("Body Style", ['sedan', 'hatchback', 'wagon', 'convertible'])
            fuel_type = st.selectbox("Fuel", ['gas', 'diesel'])
            aspiration = st.selectbox("Turbo?", ['std', 'turbo'])
            drive_wheels = st.selectbox("Traction", ['rwd', 'fwd', '4wd'])
            
        with col2:
            horsepower = st.slider("Horsepower", 50, 400, 150)
            engine_size = st.slider("Engine Size", 50, 400, 150)
            curb_weight = st.slider("Weight (lbs)", 1500, 5000, 2500)
            city_mpg = st.slider("City MPG", 10, 60, 25)
            highway_mpg = st.slider("Highway MPG", 10, 60, 30)
            
        # Defaults for other columns needed by preprocessor
        # (This is a simplified form, realistic one needs all fields or good defaults)
        
        predict_btn = st.form_submit_button("💰 Calculează Preț")
        
    if predict_btn:
        # Construct DataFrame
        # IMPORTANT: Needs to match columns expected by clean_data/feature_engineering
        input_data = pd.DataFrame([{
            'symboling': 0, 'normalized-losses': 115, # Defaults
            'make': make, 'fuel-type': fuel_type, 'aspiration': aspiration,
            'num-of-doors': 'four', 'body-style': body_style, 
            'drive-wheels': drive_wheels, 'engine-location': 'front',
            'wheel-base': 100, 'length': 170, 'width': 65, 'height': 50, # Defaults
            'curb-weight': curb_weight, 'engine-type': 'ohc', 'num-of-cylinders': 'four',
            'engine-size': engine_size, 'fuel-system': 'mpfi', 'bore': 3.0, 'stroke': 3.0,
            'compression-ratio': 9.0, 'horsepower': horsepower, 'peak-rpm': 5000,
            'city-mpg': city_mpg, 'highway-mpg': highway_mpg
        }])
        
        with st.spinner("Calculare..."):
            price = predict_price("xgboost", input_data)
        
        if price:
            st.balloons()
            
            # --- ADVANCED ANALYSIS MODULE ---
            # 1. Safety Score (Calculated in preprocessor, we re-calculate for display logic or extract if returned)
            # Since prediction returns only price, we need to inspect the processed input or recalc logic here for UI.
            # Simple UI Calculation for instant feedback:
            try:
                weight_norm = (curb_weight - 1400) / 2700
                risk_norm = 1 - ((115 - 65) / 191) # Using default losses 115
                safety_score = (0.6 * weight_norm + 0.4 * risk_norm) * 10
                safety_score = min(max(safety_score, 1), 10)
            except: safety_score = 5.0

            # 2. Classic Value Estimation (Inflation ~2.85x + Brand Premium)
            inflation = 2.85
            brand_mult = 1.5 if make in ['mercedes-benz', 'bmw', 'porsche', 'jaguar', 'alfa-romero'] else (1.1 if make in ['volvo', 'audi'] else 0.6)
            classic_val = price * inflation * brand_mult
            
            # Display Results
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px; border: 2px solid #28a745;">
                <h3 style="color: #155724; margin-bottom:0;">Preț Estimat (1985 Market)</h3>
                <h1 style="font-size: 3.5em; margin: 10px 0; color: #28a745;">${price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style="padding: 15px; background-color: #e2e3e5; border-radius: 10px; margin-top: 10px; text-align: center;">
                    <h4>🛡️ Safety Score</h4>
                    <h2 style="color: {'#28a745' if safety_score > 7 else '#ffc107'};">{safety_score:.1f}/10</h2>
                    <p style="font-size: 0.8em;">Bazat pe greutate și istoric daune</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div style="padding: 15px; background-color: #cce5ff; border-radius: 10px; margin-top: 10px; text-align: center;">
                    <h4>💰 Estimare Azi</h4>
                    <h2 style="color: #004085;">${classic_val:,.0f}</h2>
                    <p style="font-size: 0.8em;">Valoare colecție (Inflație + Brand)</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.info("💡 **Analiză AI**: Această mașină are un scor de siguranță calculat automat. Valoarea de colecție este o estimare pentru un exemplar în stare perfectă.")

elif page == "🤖 AI Assistant":
    import google.generativeai as genai
    from src.config import GEMINI_API_KEY
    from src.financial.engine import GrokModel, GPT2Model
    import os
    
    st.title("🤖 Advanced Auto Assistant")
    st.markdown("Discută cu experții noștri virtuali (Gemini, Grok sau GPT-2).")
    
    # --- MODEL SELECTION ---
    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        model_choice = st.selectbox(
            "Alege Modelul AI:",
            ["Gemini 1.5 (Google) - Online", "Grok (xAI) - Online", "GPT-2 (Local Inference) - Offline", "Simulare (Mock)"],
            index=0,
            help="Selectează 'creierul' din spatele asistentului."
        )
    
    # API Key Handling (Shared logic)
    api_key_gemini = GEMINI_API_KEY
    api_key_grok = os.getenv("GROK_API_KEY")
    
    # Status Indicators
    with col_info:
        if "Gemini" in model_choice:
            if api_key_gemini and len(api_key_gemini) > 10:
                st.success("🟢 Gemini Ready")
            else:
                st.error("🔴 Gemini Key Missing")
        elif "Grok" in model_choice:
            if api_key_grok and len(api_key_grok) > 10:
                st.success("🟢 Grok Ready")
            else:
                st.error("🔴 Grok Key Missing")
        elif "GPT-2" in model_choice:
             st.info("🟠 Model Local (CPU)")

    # Initialize Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "assistant", "content": "Salut! Cu cine vrei să vorbești astăzi?"})

    # Display Chat
    for msg in st.session_state.chat_history:
        avatar = "🤖"
        if msg["role"] == "user": avatar = "👤"
        elif "Grok" in model_choice and msg["role"] == "assistant": avatar = "🧠" 
        
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
    
    # User Input
    user_input = st.chat_input("Scrie mesajul tău...")
    
    if user_input:
        # Add to UI history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        reply_text = "Eroare generare răspuns."
        
        # --- GENERATION LOGIC ---
        
        # 1. GEMINI
        if "Gemini" in model_choice:
            if api_key_gemini:
                try:
                    genai.configure(api_key=api_key_gemini)
                    # Mega-Prompt
                    df = load_data()
                    context = f"""
                    Ești un "Senior Automotive Market & Safety Analyst" și pasionat auto.
                    Date context: {', '.join(df['make'].unique()[:10])}...
                    Răspunde prietenos și detaliat.
                    """
                    if "chat_session" not in st.session_state:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        st.session_state.chat_session = model.start_chat(history=[{"role": "user", "parts": context}])
                    
                    with st.spinner("Gemini scrie..."):
                        response = st.session_state.chat_session.send_message(user_input)
                        reply_text = response.text
                except Exception as e:
                    reply_text = f"Eroare Gemini: {e}"
            else:
                reply_text = "Lipsă cheie API Gemini (.env)."
        
        # 2. GROK
        elif "Grok" in model_choice:
            if api_key_grok:
                try:
                    grok = GrokModel(api_key=api_key_grok)
                    # Custom Persona for Chat
                    persona = """Ești un entuziast auto cu cunoștințe enciclopedice. 
                    Stilul tău este tehnic dar pasionat. Poți vorbi despre orice mașină, nu doar cele din baza de date."""
                    
                    with st.spinner("Grok gândește..."):
                         reply_text = grok.genereaza(user_input, system_instruction=persona)
                except Exception as e:
                    reply_text = f"Eroare Grok: {e}"
            else:
                reply_text = "Lipsă cheie API Grok (GROK_API_KEY in .env)."

        # 3. GPT-2
        elif "GPT-2" in model_choice:
            try:
                # Lazy load to save memory if not used
                with st.spinner("GPT-2 (Local) generează..."):
                    gpt = GPT2Model()
                    # GPT-2 is simple autocomplete, so we prompt it carefully
                    prompt = f"User asks about cars: {user_input}\nExpert Answer:"
                    reply_text = gpt.genereaza(prompt, max_length=150)
            except Exception as e:
                reply_text = f"Eroare GPT-2: {e}"

        # 4. MOCK
        else:
            from src.ai.mock_assistant import MockAI
            df = load_data()
            mock_ai = MockAI(df)
            with st.spinner("Mod Simulare..."):
                time.sleep(0.5)
                reply_text = mock_ai.generate_response(user_input)
        
        # Add to UI history
        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
        with st.chat_message("assistant", avatar="🤖" if "Grok" not in model_choice else "🧠"):
            st.write(reply_text)

elif page == "Financial Analysis":
    from src.financial.engine import FinancialCoordinator
    from src.financial.report_generator import ReportGenerator
    from src.data.companii_automotive import COMPANII_AUTOMOTIVE
    import plotly.graph_objects as go
    import tempfile

    # --- PREMIUM UI CSS ---
    st.markdown("""
    <style>
    /* Main Background adjustments if needed */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Premium Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f3f4f6;
        font-weight: 600;
    }
    
    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: ball;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Financial Analysis & AI Prediction")
    st.markdown("### Professional Stock Analysis Platform")
    
    # Initialize Coordinator
    @st.cache_resource
    def get_financial_coordinator():
        return FinancialCoordinator()
        
    coordinator = get_financial_coordinator()
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        if st.button("🔄 Actualizează Date Istorice"):
            with st.spinner("Actualizare date..."):
                coordinator.ensure_data_updated()
            st.success("Date actualizate!")
            
    # Main Search Input
    
    # Prepare dropdown options
    # Format: "Tesla (TSLA) - SUA"
    company_options = [f"{c[0]} ({c[1]})" for c in COMPANII_AUTOMOTIVE]
    # Map "Tesla (TSLA)" -> "TSLA"
    option_to_symbol = {f"{c[0]} ({c[1]})": c[1] for c in COMPANII_AUTOMOTIVE}
    
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        # Default to Tesla if available
        default_index = 0
        for i, opt in enumerate(company_options):
            if "Tesla" in opt:
                default_index = i
                break
                
        selected_option = st.selectbox("Alege Compania", company_options, index=default_index, label_visibility="collapsed")
        symbol_input = option_to_symbol[selected_option]
        
    with col_in2:
        analyze_btn = st.button("Analizează 🚀", use_container_width=True)
        
    if analyze_btn or symbol_input:
        with st.spinner(f"Processing Analysis for {selected_option}..."):
            try:
                result = coordinator.analyze_company(symbol_input)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # 1. Header & Metrics
                    st.markdown(f"## 🏢 {result['company_name']} <span style='color:#6b7280; font-size:0.8em'>({result['symbol']})</span>", unsafe_allow_html=True)
                    
                    m = result['metrics']
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Preț Curent", f"${result['current_price']:.2f}", 
                              delta=f"{m['evolutie_procent']:.2f}%")
                    c2.metric("Trend", m['trend'].upper(), 
                              delta_color="normal" if m['trend']=="stabil" else ("inverse" if m['trend']=="descrescator" else "off")) 
                    c3.metric("Volatilitate", f"{m['volatilitate']:.2f}%", 
                              delta="Risc Ridicat" if m['volatilitate'] > 3 else "Risc Scăzut")
                    c4.metric("Consensus", result['consensus'], 
                               delta=f"{result['vote_counts']['CUMPARA']} Voturi", delta_color="off")
                    
                    st.markdown("---")
                    
                    # 2. Charts & Predictions Layout
                    c_chart, c_pred = st.columns([2, 1])
                    
                    with c_chart:
                        st.subheader("📉 Market Data")
                        df = result['data']
                        
                        fig = go.Figure(data=[go.Candlestick(x=df['Data'],
                                        open=df['Open'], high=df['High'],
                                        low=df['Low'], close=df['Close'],
                                        name='OHLC')])
                                        
                        fig.update_layout(
                            xaxis_title="Data",
                            yaxis_title="Preț ($)",
                            height=450,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="#f3f4f6")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c_pred:
                        st.subheader("🤖 Model Signals")
                        preds = result['predictions']
                        
                        # Prepare data for display
                        pred_data = []
                        for model, val in preds.items():
                            if model == "Prolog_explicatie": continue
                            if val == 1: 
                                decision = "🟢 BUY"
                            elif val == -1: 
                                decision = "🔴 SELL"
                            else: 
                                decision = "🟡 HOLD"
                            pred_data.append({"Model": model, "Signal": decision})
                            
                        # Styled dataframe
                        st.dataframe(
                            pd.DataFrame(pred_data).set_index("Model"), 
                            use_container_width=True,
                            column_config={"Signal": st.column_config.TextColumn("Signal", width="medium")}
                        )
                        
                        st.info(f"**Consensus:** {result['consensus']}")

                    # 3. AI Insights & Reporting
                    st.markdown("---")
                    st.subheader("🧠 Advanced Intelligence & Reporting")
                    
                    c_ai, c_report = st.columns([3, 1])
                    
                    # AI Analysis Logic
                    gpt_text = "(Generare la cerere...)"
                    grok_text = "(Generare la cerere...)"
                    
                     # Auto-generate or just placeholder? Let's auto-generate context if needed or lazy load
                     # User explicitly asked for Grok/GPT-2 in the tab, doing it efficiently:
                    
                    with c_ai:
                        with st.expander("Show AI Analyst Insights", expanded=True):
                            with st.spinner("Consulting AI Models..."):
                                # Check if we already have it or need to call
                                # For speed, we might want to just call it
                                try:
                                   if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {}
                                   cache_key = f"{result['symbol']}_{result['data'].iloc[-1].name}"
                                   
                                   if cache_key in st.session_state.ai_cache:
                                       grok_text, gpt_text = st.session_state.ai_cache[cache_key]
                                   else:
                                       # Call engine
                                       gpt_text = coordinator.generate_gpt_analysis(result['symbol'], m)
                                       grok_text = coordinator.generate_grok_analysis(result['symbol'], m, result['consensus'])
                                       st.session_state.ai_cache[cache_key] = (grok_text, gpt_text)
                                       
                                   st.markdown(f"**Grok 🧠:** {grok_text}")
                                   st.divider()
                                   st.markdown(f"**GPT-2 🤖:** {gpt_text}")
                                except Exception as e:
                                    st.warning(f"AI Generation Error: {e}")

                    with c_report:
                        st.write("Generare Raport PDF")
                        if st.button("📄 Download PDF Report"):
                            with st.spinner("Generating PDF..."):
                                # Try to save chart
                                chart_path = None
                                try:
                                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                        fig.write_image(tmpfile.name)
                                        chart_path = tmpfile.name
                                except Exception as e:
                                    st.warning(f"Chart image export failed (Kaleido missing?), PDF will be text only. {e}")
                                
                                # Generate PDF
                                combined_ai_text = f"GROK: {grok_text}\n\nGPT-2: {gpt_text}"
                                pdf_path = ReportGenerator.generate_pdf(
                                    data=result,
                                    metrics=m,
                                    predictions=result['predictions'],
                                    consensus=result['consensus'],
                                    ai_analysis=combined_ai_text,
                                    chart_image_path=chart_path
                                )
                                
                                # Read PDF binary
                                with open(pdf_path, "rb") as f:
                                    pdf_data = f.read()
                                    
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_data,
                                    file_name=f"Report_{result['symbol']}.pdf",
                                    mime="application/pdf"
                                )
                                st.success("Ready!")

            except Exception as e:
                st.error(f"Eroare procesare: {e}")

