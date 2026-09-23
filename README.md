# Classificador de Texto para Triagem de Atendimento

Direciona automaticamente uma solicitação escrita para a área responsável, reduzindo o tempo de
triagem manual.

O projeto responde a uma pergunta objetiva: o tratamento linguístico do português melhora a
classificação, ou apenas acrescenta uma etapa ao pipeline?

## Resultado

Cinco modelos avaliados nas mesmas partições, com validação cruzada estratificada de 5 folds.

| Modelo | Macro-F1 | Desvio entre folds |
|---|---:|---:|
| NLTK + TF-IDF + Regressão Logística | 0,715 | 0,025 |
| TF-IDF + Regressão Logística | 0,685 | 0,052 |
| TF-IDF + Naive Bayes | 0,666 | 0,062 |
| NLTK + TF-IDF + Naive Bayes | 0,663 | 0,030 |
| Baseline (classe majoritária) | 0,036 | 0,000 |

Ganho sobre o baseline ingênuo: +0,680 de macro-F1.

O experimento tem duas leituras:

1. O pré-processamento com NLTK elevou o melhor modelo de 0,685 para 0,715 e reduziu o desvio
   entre folds pela metade (0,052 para 0,025). O ganho de estabilidade importa mais que o de
   média: o modelo passou a depender menos de qual partição caiu no teste.
2. O mesmo pré-processamento piorou o Naive Bayes. A normalização não é boa por princípio, ela
   interage com o classificador. Isso só fica visível porque os quatro arranjos foram medidos.

## Dado

Corpus de 140 solicitações em português, distribuídas em 7 categorias.

Limitação assumida: o corpus é pequeno e perfeitamente balanceado, com 20 documentos por
categoria. Balanceamento exato não ocorre em dado real. Os números acima demonstram que o
pipeline funciona e que a comparação entre modelos é válida, não que o modelo esteja pronto
para produção.

## Método

| Etapa | Escolha | Justificativa |
|---|---|---|
| Baseline | Classe majoritária | Piso obrigatório. Modelo que não o supera não aprendeu nada do texto. |
| Pré-processamento | NLTK: tokenização, stopwords, stemmer RSLP | RSLP é projetado para a morfologia do português |
| Representação | TF-IDF, unigrama e bigrama | Adequado a corpus pequeno |
| Classificadores | Regressão Logística, Naive Bayes Multinomial | Famílias distintas: linear e probabilística |
| Validação | Cruzada estratificada, 5 folds | Uma divisão única deixaria cerca de 28 documentos no teste, insuficiente para distinguir modelos |
| Métrica | Macro-F1 | Pesa todas as categorias igualmente |

## Análise de erro

O erro se concentra na fronteira entre `pagamento`, `fraude` e `problema_tecnico`. Exemplos
reais classificados em outra categoria:

| Texto | Rótulo | Previsto |
|---|---|---|
| "Solicitei o cancelamento do plano mas continuo sendo cobrado" | pagamento | cancelamento |
| "Recebi cobrança de um plano que nunca contratei" | pagamento | fraude |

Em ambos o texto pertence genuinamente às duas categorias. Parte relevante do erro é
ambiguidade do rótulo, não limitação do classificador. O encaminhamento correto é rótulo
múltiplo ou hierarquia de categorias, não um modelo maior.

## Auditoria de dados pessoais

Executada antes de qualquer treino. O módulo `triage.audit` varre o corpus em busca de CPF,
e-mail, telefone e cartão de pagamento, e expõe `redact()` para substituir as ocorrências por
marcadores de tipo antes que o texto alcance um modelo, um log ou uma API externa.

Neste corpus não há ocorrências, consequência de ser sintético. A varredura permanece no código
porque dado real contém esses padrões.

## Arquitetura

```
src/triage/
├── dataset.py        carregamento e propriedades do corpus
├── preprocessing.py  estratégias de normalização
├── models.py         construção dos pipelines e do baseline
├── evaluation.py     validação cruzada e consolidação dos scores
├── audit.py          dados pessoais e balanceamento
├── reporting.py      saída em console e figuras
└── __main__.py       interface de linha de comando
```

Cada módulo tem uma responsabilidade. `Preprocessor` é um `Protocol`: acrescentar uma estratégia
de normalização não exige alterar `models.py`, e os testes injetam implementações próprias sem
tocar no código de produção.

## Instalação e uso

```bash
pip install -e ".[dev]"
python -m triage
```

Opções:

```bash
python -m triage --corpus caminho/para/corpus.csv --folds 10 --output figuras/
```

## Testes

```bash
pytest
ruff check .
```

São 17 testes cobrindo carregamento, validação de entrada, comportamento do stemmer, construção
dos modelos e detecção de dados pessoais. Um deles registra uma limitação conhecida do RSLP:
o algoritmo unifica `cancelado`, `cancelamento` e `cancelar` em `cancel`, mas não unifica
`cobrar` e `cobrança`, que mantêm radicais distintos.

## Próximo passo

Substituir o corpus sintético por reclamações reais do consumidor.gov.br. O código não muda,
muda a fonte. O que muda é a qualidade da conclusão: texto com ruído real, classes
desbalanceadas de fato e dados pessoais que tornam a anonimização obrigatória em vez de
preventiva.
