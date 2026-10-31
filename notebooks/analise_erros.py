# Análise de erros do modelo publicado, sem precisar treinar de novo.
# Refaz a mesma divisão de teste do notebook de treino (mesma seed) e gera
# matriz_confusao.png e erros.csv. Rode numa célula do Colab (GPU acelera, mas não é obrigatória).
import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import torch
from google.colab import files, userdata
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline

SEED = 42
REPO_ID = "Vwhtarj/bertimbau-avaliacoes-olist"

for nome in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
    os.environ[nome] = userdata.get(nome)

# Mesma preparação do notebook de treino
pasta = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
reviews = pd.read_csv(f"{pasta}/olist_order_reviews_dataset.csv")
df = reviews[reviews["review_comment_message"].notna() & (reviews["review_score"] != 3)].copy()
df["texto"] = (
    df["review_comment_title"].fillna("") + ". " + df["review_comment_message"]
).str.strip(". ").str.replace(r"\s+", " ", regex=True).str.strip()
df["label"] = (df["review_score"] >= 4).astype(int)
df = df[df["texto"].str.len() >= 3].drop_duplicates("texto")[["texto", "label", "review_score"]].reset_index(drop=True)
_, resto = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
_, teste = train_test_split(resto, test_size=0.5, stratify=resto["label"], random_state=SEED)

clf = pipeline("text-classification", model=REPO_ID, device=0 if torch.cuda.is_available() else -1)
saidas = clf(teste["texto"].tolist(), batch_size=64, truncation=True, max_length=128)
teste = teste.assign(
    previsto=[int(s["label"] == "positivo") for s in saidas],
    prob_positivo=[round(s["score"] if s["label"] == "positivo" else 1 - s["score"], 3) for s in saidas],
)
print(f"Teste: {len(teste)} | acurácia {accuracy_score(teste['label'], teste['previsto']):.4f}"
      f" | F1 macro {f1_score(teste['label'], teste['previsto'], average='macro'):.4f}")

matriz = ConfusionMatrixDisplay.from_predictions(
    teste["label"], teste["previsto"], display_labels=["negativo", "positivo"], cmap="Greens", colorbar=False
)
matriz.ax_.set(title="BERTimbau — conjunto de teste", xlabel="Previsto", ylabel="Real")
matriz.figure_.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

erros = teste[teste["label"] != teste["previsto"]].sort_values("prob_positivo")
erros.to_csv("erros.csv", index=False)
print(f"{len(erros)} erros salvos em erros.csv")

files.download("matriz_confusao.png")
files.download("erros.csv")
