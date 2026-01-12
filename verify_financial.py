
import sys
import os
import textwrap
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.getcwd())

from src.financial.engine import (
    FinancialCoordinator, 
    BazaDate, 
    HuggingFaceNER, 
    YahooFinance, 
    DataManager, 
    ExportDate, 
    NextDayPredictor, 
    AnalizaNumPy, 
    RFModel, 
    NaiveBayesTabular, 
    ModelEvaluationOrchestrator, 
    consensus,
    COMPANII_AUTOMOTIVE
)
from src.financial.dl_models import (
    GPT2Model,
    GrokModel,
    NNModel,
    LSTMModel,
    GRUModel,
    TCNModel,
    TransformerModel
)

# AFISARE HEADER ȘI ISTORIC EVOLUȚIE MODELE AI
def afiseaza_istoric_AI():
    print("=" * 44)
    print("SISTEM INTEGRAT DE ANALIZĂ FINANCIARĂ AI")
    print("De la NaiveBayes (1763) la Grok AI (2024)")
    print("=" * 44)
    print("\nIstoric evoluție modele utilizate în ML/AI:\n")

    istoric = [
        (1763, "NaiveBayes", "Statistic / ML"),
        (1906, "MarkovChain", "Statistic / AI secvențial"),
        (1972, "Prolog", "AI simbolic / logic"),
        (1986, "NeuralNetwork", "Deep Learning / MLP"),
        (1997, "LSTM", "Deep Learning / AI secvențial"),
        (2001, "RandomForest", "ML clasic"),
        (2014, "GRU", "Deep Learning / AI secvențial"),
        (2016, "TCN", "Deep Learning / AI secvențial"),
        (2017, "Transformer", "Deep Learning modern / AI"),
    ]

    for an, descriere, tip in istoric:
        print(f"{an} ─ {descriere} [{tip}]")

def main():
    load_dotenv()
    grok_api_key = os.getenv("GROK_API_KEY")
    
    afiseaza_istoric_AI()
    
    hf = HuggingFaceNER(COMPANII_AUTOMOTIVE)
    # Listează companiile disponibile
    print("\nCompanii disponibile în sistem:")
    for nume, simbol in hf.simboluri.items():
        print(f"  - {nume.title()}: {simbol}")
        
    db = BazaDate()
    ner = HuggingFaceNER(COMPANII_AUTOMOTIVE)
    yf_client = YahooFinance()
    data_manager = DataManager(db, ner, yf_client)
    
    # Check data
    data_manager.verifica_si_descarca_date(prag_min_randuri=200)

    exporter = ExportDate(db)
    predictor = NextDayPredictor(window=30, n_lags=5)
    analizator_numpy = AnalizaNumPy()

    intrebari = [
        "Care este evaluarea financiară a firmei TESLA azi?",
    ]

    ROLLING_WINDOW = 900
    label_mapping = {-1: "VINDE", 0: "PASTREAZA", 1: "CUMPARA"}

    for intrebare in intrebari:
        print(f"\nÎntrebare: {intrebare}")

        # Identificare Companie (NER)
        nume_companie, simbol = ner.extrage_companie(intrebare)

        # Export Date și Info Fișier
        df, filehash = exporter.exporta(simbol, zile=ROLLING_WINDOW)
        if df is None or df.empty:
            print(f"Nu există date pentru simbolul {simbol}.")
            # Try to force download if missing
            if data_manager.asigura_date_pentru_simbol(simbol):
                 df, filehash = exporter.exporta(simbol, zile=ROLLING_WINDOW)
            
            if df is None or df.empty:
                 continue

        print(f"\n  Date disponibile: {len(df)} zile ({df['Data'].iloc[0]} → {df['Data'].iloc[-1]})")
        print(f"  CSV exportat: date_per_simbol/{simbol}_{ROLLING_WINDOW}_zile.csv")

        # HUGGING FACE (NER)
        print(f"  HUGGING FACE (NER) → Companie: {nume_companie}, Simbol: {simbol}")

        # YAHOO FINANCE (Date brute)
        pret_azi = float(df["Close"].iloc[-1])
        pret_ieri = float(df["Close"].iloc[-2])
        print(f"  YAHOO FINANCE → Preț ieri: ${pret_ieri:.2f}, Preț azi: ${pret_azi:.2f}")

        # NUMPY ANALYSIS (Calcul metrici statistice)
        metrici = analizator_numpy.calculeaza_metrici({
            "pret_ieri": pret_ieri,
            "pret_azi": pret_azi,
            "istoric_complet": df
        })

        print(f"  NUMPY ANALYSIS → Evoluție: {metrici['evolutie_procent']:+.2f}%, "
              f"Volatilitate: {metrici['volatilitate']:.2f}%, "
              f"Momentum: {metrici['momentum']:.2f}, "
              f"Trend: {metrici['trend']}")

        rezultate = predictor.ruleaza(df)
        pret_curent = df["Close"].iloc[-1]

        print("\nPredicții pentru ziua următoare (pe baza istoricului):")
        for model, pred in rezultate.items():
            if model == "Prolog_explicatie":
                continue

            if pred is None:
                print(f"  {model}: NECUNOSCUT")
            else:
                try:
                    p_val = int(pred)
                    print(f"  {model}: {label_mapping.get(p_val, 'NECUNOSCUT')}")
                except:
                    print(f"  {model}: {pred}")

        decizie_finala, rezultate_clase = consensus(rezultate, pret_curent)

        if decizie_finala is None:
            print("\n⚠Nu s-a putut lua o decizie (prea puține predicții).")
        else:
            toate_voturile = list(rezultate_clase.values())

            nr_cumpara = toate_voturile.count(1)
            nr_vinde = toate_voturile.count(-1)
            nr_pastreaza = toate_voturile.count(0)

            print(f"\nCUMPĂRĂ = {nr_cumpara}, VINDE = {nr_vinde}, PĂSTREAZĂ = {nr_pastreaza}.")
            print(f"Decizie finală (vot majoritar între modele): {label_mapping.get(decizie_finala, 'NECUNOSCUT')}")

        model_gpt2 = GPT2Model()
        model_grok = GrokModel(api_key=grok_api_key)

        print("\nGPT-2 PREDICTION WORKFLOW")
        context_gpt2 = f"Stock {simbol} moved {metrici['evolutie_procent']:.2f}% today."
        try:
             # Run debug gen for effect
             model_gpt2.genereaza(context_gpt2, max_length=200)
        except Exception as e:
             print(f"GPT-2 Debug Error: {e}")

        print(f"\n • Grok AI Analysis:")
        d_fin = label_mapping.get(decizie_finala, "NECUNOSCUT")
        context_grok = f"Analizează simbolul {simbol}. Evoluție: {metrici['evolutie_procent']}%, Trend: {metrici['trend']}. Decizie consensus: {d_fin}."
        grok_text = model_grok.genereaza(context_grok)

        grok_linii = textwrap.wrap(grok_text, width=100)
        for linie in grok_linii[:5]:
            print(f"  {linie}")
        if len(grok_linii) > 5:
            print(f"  [...] (text continuă)")

    print("DEMARARE EVALUARE AUTOMATĂ MODELE")
    print("SISTEM COMPLET DE EVALUARE PERFORMANȚĂ MODELE ML/DL")
    
    print("\n[1/4] Calculare metrici de performanță...")
    # Using orchestrator logic directly/indirectly
    
    X_eval, y_true_eval = predictor.tab_builder.construieste_dataset(df)

    if len(X_eval) > 60:
        X_test = X_eval[-50:]
        y_true = y_true_eval[-50:]
        X_train_eval = X_eval[:-50]
        y_train_eval = y_true_eval[:-50]

        model_rf = RFModel()
        model_rf.train(X_train_eval, y_train_eval)

        model_nb = NaiveBayesTabular()
        model_nb.train(X_train_eval, y_train_eval)

        model_nn = NNModel(input_dim=X_eval.shape[1])
        model_nn.train(X_train_eval, y_train_eval)

        dict_predictii = {
            "RandomForest": model_rf.predict(X_test),
            "NaiveBayes": model_nb.predict(X_test),
            "NeuralNetwork": model_nn.predict(X_test)
        }

        orchestrator = ModelEvaluationOrchestrator()

        metrici_finale = orchestrator.evalueaza_complet(
            y_true=y_true,
            predictions_dict=dict_predictii,
            output_dir=f"evaluare_{simbol}"
        )
    else:
        print("Nu sunt suficiente date pentru evaluare completă (necesar > 60 zile).")

if __name__ == "__main__":
    main()
