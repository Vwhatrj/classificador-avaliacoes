# Classificador de avaliações em português

Fine-tuning do **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) para classificar avaliações de compras como **positivas** ou **negativas**, com dados reais do e-commerce brasileiro ([Olist, Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)). Tudo roda de graça no Google Colab, e o modelo final fica publicado no Hugging Face.

<a href="https://huggingface.co/spaces/Vwhtarj/classificador-avaliacoes"><img src="docs/demo.png" alt="Demo: a avaliação 'Comprei faz um mês e até agora nada. Péssimo.' classificada como negativa com 98,9%" width="480"></a>

- **Teste no navegador:** [demo no Hugging Face Spaces](https://huggingface.co/spaces/Vwhtarj/classificador-avaliacoes). Roda no seu navegador com Transformers.js, sem servidor ([`demo/index.html`](demo/index.html)).
- **Modelo publicado:** [Vwhtarj/bertimbau-avaliacoes-olist](https://huggingface.co/Vwhtarj/bertimbau-avaliacoes-olist)
- **Notebook:** [`notebooks/treino_bertimbau_olist.ipynb`](notebooks/treino_bertimbau_olist.ipynb)

## Resultados (conjunto de teste)

3.309 avaliações nunca vistas no treino. Os dados somam 33.083 avaliações, das quais 68,6% são positivas.

| Modelo | Acurácia | F1 macro |
|---|---|---|
| TF-IDF + Regressão Logística (baseline) | 0,9293 | 0,9206 |
| **BERTimbau fine-tuned** | **0,9516** | **0,9442** |

Em relação ao baseline, o BERTimbau erra cerca de **30% menos**: a taxa de erro caiu de 7,1% para 4,8%. O treino levou 2 épocas numa GPU T4 gratuita do Colab.

## Decisões

- **Rótulos:** nota 1–2 = negativo, 4–5 = positivo. A nota 3 fica de fora porque é ambígua.
- **Sem vazamento:** textos duplicados são removidos antes de dividir. O teste estratificado de 10% é usado uma única vez.
- **Baseline primeiro:** o TF-IDF + Regressão Logística mostra quanto o BERT realmente acrescenta.
- **Métrica:** como as classes são desbalanceadas, a principal é o F1 macro, não a acurácia.

## Como reproduzir

1. Crie as contas e os tokens:
   - no [Kaggle](https://www.kaggle.com/settings): *Create New Token* gera `KAGGLE_USERNAME` e `KAGGLE_KEY`;
   - no [Hugging Face](https://huggingface.co/settings/tokens): um token com permissão **Write** vira `HF_TOKEN`.
2. Abra o notebook no [Google Colab](https://colab.research.google.com/) (*Arquivo → Fazer upload de notebook*).
3. Escolha *Ambiente de execução → Alterar o tipo → GPU T4*.
4. Cadastre os três valores em **Secrets** (ícone de chave na barra lateral) e ligue o "Acesso ao notebook".
5. Rode *Ambiente de execução → Executar tudo*. O treino leva uns 15–25 minutos na T4.

No fim, o notebook publica o modelo e o model card em `seu-usuario/bertimbau-avaliacoes-olist` e grava `resultados.json` com as métricas.

## Uso do modelo publicado

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Vwhtarj/bertimbau-avaliacoes-olist")
clf("Chegou rápido e funciona perfeitamente!")
```

## Limitações

O modelo foi treinado só com avaliações de e-commerce e é binário, sem classe neutra. Ironia continua difícil, e a nota dada pelo cliente é um rótulo ruidoso. Os detalhes estão no model card.
