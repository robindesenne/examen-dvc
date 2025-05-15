# Examen DVC / Dagshub – Robin DESENNE

Ce dépôt contient la solution complète à l’examen :

| Élément | Chemin |
|---------|--------|
| Données brutes | `data/raw/raw.csv` |
| Données traitées | `data/processed/…` |
| Modèle entraîné | `models/models/gbr_model.pkl` |
| Métriques | `metrics/scores.json` |
| Pipeline DVC | `dvc.yaml` + `dvc.lock` |

Le graphe de pipeline, les datasets et le modèle sont visibles directement sur Dagshub.

## Lien Dagshub

🔗 <https://dagshub.com/desenne.robin/examen-dvc>

*(Lecteur « Licence Pédagogique » ajouté – read-only)*

## Lancer la pipeline en local

```bash
git clone https://github.com/robindesenne/examen-dvc.git
cd examen-dvc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull        # récupère données + modèle
dvc repro       # rejoue la pipeline
