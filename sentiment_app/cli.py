import argparse, sys, os, glob

def find_files():
    base = os.getcwd()
    models_dir = os.path.join(base, "models")
    joblibs = glob.glob(os.path.join(models_dir, "*.joblib"))
    model = [f for f in joblibs if "model" in os.path.basename(f).lower()]
    vec = [f for f in joblibs if "vec" in os.path.basename(f).lower() or "tfidf" in os.path.basename(f).lower()]
    if model and vec:
        return model[0], vec[0]
    if model:
        return model[0], None
    if joblibs:
        return joblibs[0], None
    return None, None

def main():
    parser = argparse.ArgumentParser(prog="sentiment", description="Analyse de sentiment v0.6.0")
    sub = parser.add_subparsers(dest="commande")
    sub.add_parser("train", help="Entraîne le modèle")
    p_pred = sub.add_parser("predict", help="Prédit le sentiment d'un texte")
    p_pred.add_argument("texte", help="Le texte à analyser")
    sub.add_parser("version", help="Affiche la version")
    args = parser.parse_args()

    if args.commande == "version":
        from . import __version__
        print(f"sentiment v{__version__}")
    elif args.commande == "train":
        print("Entraînement du modèle...")
        os.system(f"python \"{os.path.join(os.getcwd(), 'scripts', 'train_model.py')}\"")
        print("Modèle entraîné → models/")
    elif args.commande == "predict":
        import joblib
        model_path, vec_path = find_files()
        if not model_path:
            print(f"Erreur : aucun .joblib dans {os.path.join(os.getcwd(), 'models')}")
            print(f"Fichiers trouvés : {os.listdir('models') if os.path.isdir('models') else 'dossier absent'}")
            sys.exit(1)
        print(f"Modèle chargé : {os.path.basename(model_path)}")
        if vec_path:
            vec = joblib.load(vec_path)
            model = joblib.load(model_path)
            X = vec.transform([args.texte])
            proba = model.predict_proba(X)[0]
            label = model.classes_[proba.argmax()]
        else:
            model = joblib.load(model_path)
            proba = model.predict_proba([args.texte])[0]
            label = model.classes_[proba.argmax()]
        score = float(proba.max())
        verdict = "POSITIF" if str(label).lower() in ["positive","pos","1","positif"] else "NEGATIF"
        print(f"{verdict} (confiance {score:.1%})")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()