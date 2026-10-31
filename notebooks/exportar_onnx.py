# Exporta o modelo publicado para ONNX (usado pela demo com Transformers.js)
# e envia para a pasta onnx/ do mesmo repositório no Hugging Face.
# Rode numa célula do Colab (não precisa de GPU). Requer o secret HF_TOKEN.
!pip install -q onnx onnxruntime

import os

import numpy as np
import onnxruntime as ort
import torch
from google.colab import userdata
from huggingface_hub import HfApi, login
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

login(token=userdata.get("HF_TOKEN"))
repo_id = f"{HfApi().whoami()['name']}/bertimbau-avaliacoes-olist"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
modelo = AutoModelForSequenceClassification.from_pretrained(repo_id).eval()


class SoLogits(torch.nn.Module):
    """Embrulha o modelo para a saída ser só o tensor de logits."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


os.makedirs("onnx", exist_ok=True)
entrada = tokenizer("Chegou rápido e bem embalado.", return_tensors="pt")
nomes = ["input_ids", "attention_mask", "token_type_ids"]
eixos = {nome: {0: "batch", 1: "sequencia"} for nome in nomes}
eixos["logits"] = {0: "batch"}

torch.onnx.export(
    SoLogits(modelo),
    tuple(entrada[n] for n in nomes),
    "onnx/model.onnx",
    input_names=nomes,
    output_names=["logits"],
    dynamic_axes=eixos,
    opset_version=14,
    dynamo=False,
)
# Versão 8 bits: ~4x menor, é a que a demo baixa no navegador
quantize_dynamic("onnx/model.onnx", "onnx/model_quantized.onnx", weight_type=QuantType.QUInt8)

# Confere se o ONNX dá o mesmo resultado que o PyTorch
with torch.no_grad():
    esperado = torch.softmax(modelo(**entrada).logits, -1).numpy()
for arquivo in ("onnx/model.onnx", "onnx/model_quantized.onnx"):
    sessao = ort.InferenceSession(arquivo)
    logits = sessao.run(None, {n: entrada[n].numpy() for n in nomes})[0]
    obtido = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    print(f"{arquivo}: {os.path.getsize(arquivo) / 1e6:.0f} MB, diferença máx. {abs(obtido - esperado).max():.4f}")

HfApi().upload_folder(folder_path="onnx", path_in_repo="onnx", repo_id=repo_id, commit_message="Versão ONNX para Transformers.js")
print(f"Enviado: https://huggingface.co/{repo_id}/tree/main/onnx")
