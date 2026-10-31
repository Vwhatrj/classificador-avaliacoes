"""Demo do classificador de avaliações (Hugging Face Spaces + Gradio)."""
import gradio as gr
from transformers import pipeline

MODELO = "Vwhtarj/bertimbau-avaliacoes-olist"
REPO_GITHUB = "https://github.com/Vwhatrj/classificador-avaliacoes"

classificador = pipeline("text-classification", model=MODELO, top_k=None)


def classificar(texto):
    texto = (texto or "").strip()
    if not texto:
        return {}
    resultados = classificador(texto[:2000], truncation=True, max_length=128)
    if resultados and isinstance(resultados[0], list):  # algumas versões aninham a lista
        resultados = resultados[0]
    return {r["label"]: float(r["score"]) for r in resultados}


demo = gr.Interface(
    fn=classificar,
    inputs=gr.Textbox(label="Avaliação", placeholder="Ex.: Chegou rápido e bem embalado!", lines=4),
    outputs=gr.Label(label="Sentimento"),
    title="Classificador de avaliações em português",
    description=(
        "BERTimbau ajustado com 26 mil avaliações reais de e-commerce (Olist). "
        "Acurácia de 95% e F1 macro de 0,94 no conjunto de teste. "
        f"[Modelo](https://huggingface.co/{MODELO}) · [Código]({REPO_GITHUB})"
    ),
    examples=[
        ["Produto chegou antes do prazo e muito bem embalado. Recomendo!"],
        ["Comprei faz um mês e até agora nada. Péssimo."],
        ["Veio com defeito, mas a loja trocou rapidinho."],
        ["Não é ruim."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
