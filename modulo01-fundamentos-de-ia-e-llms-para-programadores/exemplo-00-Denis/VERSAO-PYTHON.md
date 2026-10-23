# Versão Python do exemplo de classificação

Este roteiro apresenta em Python o mesmo exemplo de `index.js`, para comparar as etapas e os conceitos entre JavaScript com TensorFlow.js e Python com TensorFlow/Keras.

> **Importante:** o programa é didático. Ele treina com apenas três exemplos; por isso, as previsões não são confiáveis para decisões reais.

## 1. Instalar e executar

Use uma versão do Python compatível com a versão do TensorFlow disponível para seu sistema. No terminal, instale TensorFlow:

```powershell
python -m pip install tensorflow
```

Salve o programa abaixo como `exemplo_tensorflow.py` e execute:

```powershell
python exemplo_tensorflow.py
```

Se a instalação falhar, verifique a compatibilidade da versão do Python e do sistema operacional com a versão do TensorFlow escolhida.

## 2. Programa completo

```python
import tensorflow as tf


def train_model(input_xs, output_ys):
    # Modelo sequencial: os dados passam pelas camadas em ordem.
    model = tf.keras.Sequential([
        # Cada exemplo de entrada tem sete características.
        tf.keras.layers.Input(shape=(7,)),

        # Camada oculta com 80 unidades e ativação ReLU.
        tf.keras.layers.Dense(80, activation="relu"),

        # Três saídas, uma para cada categoria.
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    # Configura como o modelo será treinado.
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Equivale ao callback onEpochEnd do exemplo em JavaScript:
    # imprime a perda ao final de cada época.
    class LossPrinter(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(f"Epoch: {epoch}: loss = {logs.get('loss')}")

    # Treina por 100 épocas, embaralhando os exemplos.
    model.fit(
        input_xs,
        output_ys,
        epochs=100,
        shuffle=True,
        verbose=0,
        callbacks=[LossPrinter()],
    )

    return model


def predict(model, pessoa):
    # Keras espera um lote de exemplos: cada linha tem sete características.
    pred_array = model.predict(pessoa, verbose=0)

    # Devolve cada probabilidade junto ao índice da categoria.
    return [
        {"prob": float(prob), "index": index}
        for index, prob in enumerate(pred_array[0])
    ]


# Ordem das características:
# [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
tensor_pessoas_normalizado = [
    [0.33, 1, 0, 0, 1, 0, 0],  # Erick: azul, São Paulo
    [0,    0, 1, 0, 0, 1, 0],  # Ana: vermelha, Rio
    [1,    0, 0, 1, 0, 0, 1],  # Carlos: verde, Curitiba
]

# A posição de cada nome corresponde ao índice da saída do modelo.
labels_nomes = ["premium", "medium", "basic"]

# Rótulos corretos, em formato one-hot:
# premium = [1, 0, 0], medium = [0, 1, 0], basic = [0, 0, 1]
tensor_labels = [
    [1, 0, 0],  # Erick: premium
    [0, 1, 0],  # Ana: medium
    [0, 0, 1],  # Carlos: basic
]

# Converte as listas para tensores do TensorFlow.
input_xs = tf.convert_to_tensor(tensor_pessoas_normalizado, dtype=tf.float32)
output_ys = tf.convert_to_tensor(tensor_labels, dtype=tf.float32)

# Treina a rede neural.
model = train_model(input_xs, output_ys)

# Descrição da pessoa do exemplo original.
pessoa = {
    "nome": "zé",
    "idade": 28,
    "cor": "verde",
    "localizacao": "Curitiba",
}

# Este vetor reproduz o vetor atualmente usado no index.js:
# idade 0.2, azul e São Paulo.
# Portanto, ele NÃO corresponde à descrição de Zé acima.
pessoa_tensor_normalizado = [
    [0.2, 1, 0, 0, 1, 0, 0]
]

# Para representar a descrição de Zé (verde e Curitiba), use em vez disso:
# pessoa_tensor_normalizado = [
#     [0.2, 0, 0, 1, 0, 0, 1]
# ]

# Calcula as probabilidades para a pessoa.
predictions = predict(model, pessoa_tensor_normalizado)

# Ordena da maior para a menor probabilidade e prepara o texto de saída.
results = sorted(predictions, key=lambda item: item["prob"], reverse=True)
results = "\n".join(
    f"{labels_nomes[item['index']]} ({item['prob'] * 100:.2f}%)"
    for item in results
)

print(results)
```

## 3. Roteiro de execução

### 3.1 Importar TensorFlow

```python
import tensorflow as tf
```

Importa a biblioteca TensorFlow e a chama de `tf` no restante do programa. Keras, acessado como `tf.keras`, fornece a API para criar, treinar e usar o modelo.

### 3.2 Criar e treinar o modelo

A função `train_model(input_xs, output_ys)` recebe as características e as categorias conhecidas. `tf.keras.Sequential` organiza as camadas em sequência:

1. `Input(shape=(7,))` declara que cada exemplo contém sete números.
2. `Dense(80, activation="relu")` cria a camada oculta com 80 unidades.
3. `Dense(3, activation="softmax")` cria uma saída para cada categoria e converte os valores em probabilidades.

`model.compile(...)` configura o otimizador Adam, a perda `categorical_crossentropy` e a métrica `accuracy`. A acurácia é calculada durante o treinamento, mas o programa imprime somente a perda.

`model.fit(...)` executa o treinamento:

- `epochs=100`: percorre os exemplos 100 vezes;
- `shuffle=True`: embaralha os exemplos;
- `verbose=0`: desativa o relatório padrão do Keras;
- `callbacks=[LossPrinter()]`: imprime a perda no final de cada época.

A função devolve `model`, agora treinado.

### 3.3 Representar os dados

Cada pessoa é representada por sete números na mesma ordem: idade normalizada, três indicadores de cor e três indicadores de localização. Para cor e localização, o `1` marca a opção selecionada e os demais valores são `0` (codificação one-hot).

`tensor_pessoas_normalizado` guarda as três entradas. `tensor_labels` guarda as respostas esperadas na mesma ordem: Erick é `premium`, Ana é `medium` e Carlos é `basic`. Os rótulos também estão em formato one-hot.

`tf.convert_to_tensor(..., dtype=tf.float32)` converte as listas em tensores de números decimais, que são passados ao modelo.

### 3.4 Preparar a pessoa e prever

O dicionário `pessoa` descreve Zé, mas não é usado diretamente pelo modelo. O modelo recebe `pessoa_tensor_normalizado`, uma matriz numérica com uma linha e sete características.

O vetor reproduzido do `index.js` atual é `[0.2, 1, 0, 0, 1, 0, 0]`: ele representa azul e São Paulo, apesar de o dicionário dizer verde e Curitiba. O código inclui, comentada, a alternativa `[0.2, 0, 0, 1, 0, 0, 1]`, que corresponde à descrição de Zé.

`predict(model, pessoa)` chama `model.predict` e associa cada probabilidade ao índice da categoria. A ordenação final usa `sorted(..., reverse=True)` para listar primeiro a maior probabilidade; `labels_nomes` converte o índice no nome, e `:.2f` formata a porcentagem com duas casas decimais.

## 4. Correspondência entre JavaScript e Python

| JavaScript (TensorFlow.js) | Python (TensorFlow/Keras) | Papel |
|---|---|---|
| `tf.sequential()` | `tf.keras.Sequential(...)` | Cria o modelo em sequência |
| `tf.layers.dense(...)` | `tf.keras.layers.Dense(...)` | Adiciona uma camada densa |
| `model.compile(...)` | `model.compile(...)` | Configura otimizador, perda e métricas |
| `model.fit(...)` | `model.fit(...)` | Treina o modelo |
| `tf.tensor2d(...)` | `tf.convert_to_tensor(...)` | Converte dados para tensores |
| `model.predict(...)` | `model.predict(...)` | Calcula as saídas para novos dados |
| `onEpochEnd` | `Callback.on_epoch_end` | Executa código ao fim de cada época |

## 5. Comparando resultados

As duas versões usam a mesma arquitetura, os mesmos exemplos, as mesmas categorias e o mesmo vetor de previsão. Mesmo assim, as porcentagens podem ser diferentes entre execuções ou entre bibliotecas: os parâmetros iniciais do modelo e detalhes de execução podem variar. O objetivo da comparação é observar como o mesmo fluxo — preparar dados, treinar e prever — é expresso em JavaScript e Python, não obter porcentagens idênticas.
