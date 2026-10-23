# Roteiro do exemplo de classificação com TensorFlow.js

Este documento explica, em ordem de execução, o que cada parte de `index.js` faz. O exemplo cria uma rede neural simples, treina-a com três registros fictícios e pede uma previsão para uma pessoa.

> **Importante:** este é um exemplo didático. Três registros de treinamento não são suficientes para produzir previsões confiáveis.

## 1. Dependência e formato de módulos

No começo de `index.js`:

```js
import tf from '@tensorflow/tfjs-node';
```

- `import` carrega o TensorFlow.js para Node.js, instalado como `@tensorflow/tfjs-node`.
- O pacote fornece a API do TensorFlow.js e um backend nativo para realizar os cálculos no computador.
- Para que o Node.js aceite a sintaxe `import` neste projeto, o `package.json` deve conter `"type": "module"`.

## 2. A função `trainModel`

```js
async function trainModel(inputXs, outputYs) {
```

Esta função monta, configura e treina a rede neural.

- `inputXs` contém as características de entrada dos exemplos.
- `outputYs` contém a categoria correta de cada exemplo.
- `async` permite usar `await` para esperar o término do treinamento.

### 2.1 Cria um modelo sequencial

```js
const model = tf.sequential()
```

Um modelo sequencial é uma pilha de camadas: os dados entram na primeira camada, seguem pela próxima e terminam na camada de saída.

### 2.2 Adiciona a camada de entrada e a camada oculta

```js
model.add(tf.layers.dense({
    inputShape: [7],
    units: 80,
    activation: 'relu'
}))
```

- `inputShape: [7]`: cada exemplo deve chegar como sete números.
- `units: 80`: a camada tem 80 unidades (neurônios), com parâmetros que serão ajustados durante o treino.
- `activation: 'relu'`: aplica a função ReLU às saídas da camada, substituindo valores negativos por zero.
- `dense`: indica uma camada em que cada unidade se conecta às entradas da camada anterior.

As sete posições representam, nesta ordem:

1. idade normalizada;
2. indicador da cor azul;
3. indicador da cor vermelha;
4. indicador da cor verde;
5. indicador da localização São Paulo;
6. indicador da localização Rio;
7. indicador da localização Curitiba.

### 2.3 Adiciona a camada de saída

```js
model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
}))
```

- `units: 3`: produz um valor para cada uma das três categorias.
- `softmax`: converte os três valores em probabilidades que somam aproximadamente 1 (ou 100%).
- A ordem das saídas é a mesma de `labelsNomes`: `premium`, `medium`, `basic`.

### 2.4 Configura como o modelo aprende

```js
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
})
```

- `optimizer: 'adam'`: escolhe o algoritmo que ajusta os parâmetros da rede durante o treino.
- `loss: 'categoricalCrossentropy'`: mede a diferença entre a saída prevista e a categoria correta, que está em formato one-hot.
- `metrics: ['accuracy']`: solicita que a acurácia seja acompanhada como métrica. O código não imprime essa métrica diretamente.

### 2.5 Treina a rede

```js
await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
        onEpochEnd: (epoch, log) => console.log(
            `Epoch: ${epoch}: loss = ${log.loss}`
        )
    }
})
```

- `model.fit(...)` treina o modelo relacionando cada linha de `inputXs` à linha correspondente de `outputYs`.
- `await` espera o treinamento terminar antes de continuar.
- `verbose: 0` desativa o relatório padrão do TensorFlow.js.
- `epochs: 100` faz o modelo percorrer os exemplos de treinamento 100 vezes.
- `shuffle: true` embaralha os exemplos durante o treinamento.
- `onEpochEnd` imprime o valor da função de perda ao fim de cada época. A numeração normalmente começa em `0`, então os registros vão de `Epoch: 0` a `Epoch: 99`.

Ao terminar, `return model` devolve o modelo já treinado.

## 3. A função `predict`

```js
async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}
```

Esta função recebe o modelo treinado e uma matriz de características:

1. `tf.tensor2d(pessoa)` converte a matriz JavaScript num tensor bidimensional, formato esperado pelo modelo. A dimensão externa é o lote de exemplos; a dimensão interna contém os sete valores de cada exemplo.
2. `model.predict(tfInput)` calcula as três probabilidades para cada linha recebida.
3. `await pred.array()` converte o tensor da previsão para uma matriz JavaScript.
4. `predArray[0]` seleciona a previsão do primeiro exemplo do lote.
5. `map` transforma cada probabilidade num objeto `{ prob, index }`. O índice serve para associá-la ao nome da categoria.

## 4. Os dados de treinamento

### 4.1 Entradas

```js
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0],
    [0,    0, 1, 0, 0, 1, 0],
    [1,    0, 0, 1, 0, 0, 1]
]
```

Cada linha é um exemplo, na ordem definida anteriormente:

| Exemplo comentado no código | Idade normalizada | Cor | Localização |
|---|---:|---|---|
| Erick | `0.33` | azul | São Paulo |
| Ana | `0` | vermelha | Rio |
| Carlos | `1` | verde | Curitiba |

As cores e localizações usam codificação one-hot: a opção representada pelo `1` está selecionada e as demais ficam em `0`. A idade também é convertida para um valor numérico entre 0 e 1. Nos comentários do exemplo, a normalização usa a fórmula `(idade - 25) / (40 - 25)`.

### 4.2 Nomes e respostas corretas

```js
const labelsNomes = ['premium', 'medium', 'basic']
const tensorLabels = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
```

`labelsNomes` define a correspondência entre índice e categoria:

| Índice | Categoria | Vetor one-hot |
|---:|---|---|
| 0 | `premium` | `[1, 0, 0]` |
| 1 | `medium` | `[0, 1, 0]` |
| 2 | `basic` | `[0, 0, 1]` |

Cada linha de `tensorLabels` representa a categoria que o modelo deve aprender para a linha de entrada correspondente. O código associa Erick a `premium`, Ana a `medium` e Carlos a `basic`.

### 4.3 Converte as matrizes em tensores

```js
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)
```

Essas chamadas convertem os arrays JavaScript em tensores bidimensionais para o TensorFlow.js. Há três exemplos em cada tensor, e as linhas correspondentes formam os pares de treinamento.

## 5. Treina o modelo

```js
const model = await trainModel(inputXs, outputYs)
```

Esta linha chama a função descrita anteriormente. A execução só continua depois das 100 épocas de treinamento; `model` passa a guardar a rede já treinada.

## 6. Define a pessoa a classificar

```js
const pessoa = {
    nome: 'zé',
    idade: 28,
    cor: 'verde',
    localizacao: 'Curitiba'
}
```

Este objeto registra a descrição da pessoa, mas **não é passado para a função `predict`**. A previsão usa a matriz `pessoaTensorNormalizado` que vem logo depois.

## 7. Monta a entrada da previsão

```js
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,   // cor azul
        0,   // cor vermelha
        0,   // cor verde
        0,   // São Paulo
        1,   // Rio
        0    // Curitiba
    ]
]
```

A matriz tem dois níveis de colchetes porque contém um lote com uma pessoa, cuja linha possui sete características.

**Atenção: há uma inconsistência no código atual.** O objeto `pessoa` diz que Zé tem cor verde e mora em Curitiba, mas os indicadores do vetor dizem azul e Rio:

- `[1, 0, 0]` nas posições de cor significa azul;
- `[0, 1, 0]` nas posições de localização significa Rio.

Para a matriz representar idade normalizada `0.2`, cor verde e Curitiba, ela deveria ser:

```js
const pessoaTensorNormalizado = [
    [0.2, 0, 0, 1, 0, 0, 1]
]
```

O vetor deve respeitar exatamente a mesma ordem das sete características usada no treinamento. A idade `0.2` corresponde à fórmula `(28 - 25) / (40 - 25)`.

## 8. Calcula e exibe as probabilidades

```js
const predictions = await predict(model, pessoaTensorNormalizado)
```

A função faz a previsão com o vetor informado. O resultado consiste em três pares de índice e probabilidade.

```js
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)
```

Em sequência:

1. `sort` ordena as categorias da maior probabilidade para a menor.
2. `labelsNomes[p.index]` converte o índice de cada saída no nome da categoria.
3. `p.prob * 100` transforma a probabilidade numa porcentagem.
4. `toFixed(2)` apresenta duas casas decimais.
5. `join('\n')` coloca cada categoria numa linha.
6. `console.log(results)` imprime as três categorias no terminal; a primeira é a de maior probabilidade.

Por exemplo, `medium (57.09%)` significa que `medium` recebeu a maior probabilidade calculada nessa execução. Não é uma garantia nem uma medida de certeza calibrada.

## 9. Por que o resultado pode mudar

O modelo inicia com parâmetros que podem variar entre execuções, e o código não define uma semente aleatória. Assim, as porcentagens podem mudar quando o programa roda novamente. Além disso:

- o treinamento tem somente três exemplos;
- 80 unidades ocultas são muitas em comparação com essa quantidade de exemplos;
- as categorias são definidas pelos rótulos fornecidos, não inferidas automaticamente dos nomes, cores ou cidades;
- a previsão depende do vetor numérico, não do objeto `pessoa`;
- porcentagens do softmax sempre são distribuídas entre as três categorias, mesmo quando o modelo não tem dados suficientes para uma resposta confiável.

O objetivo do programa é demonstrar o processo de converter dados em números, treinar uma rede e obter probabilidades para uma classificação.

## 10. Executar o exemplo

Na pasta do projeto, com a dependência instalada e a versão de Node.js usada para preparar o ambiente, execute:

```powershell
npm start
```

O script `start` configurado no `package.json` executa `node --no-warnings --watch index.js`. A opção `--watch` faz o Node observar o arquivo e executá-lo novamente quando ele muda. Para parar o processo no terminal, pressione `Ctrl+C`.
