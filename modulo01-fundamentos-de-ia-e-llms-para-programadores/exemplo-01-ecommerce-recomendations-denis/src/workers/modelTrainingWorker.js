import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

//Normalizar valores contínuos (preço, idade) para o intervalo de 0 a 1
//Por quê? Mantém todos os recursos balanceados para que nenhum domine o treinamento
//Fórmula: (val min) / (max min)
//Exemplo: preço=129,99, preçoMínimo=39,99, preçoMáximo=199,99 0,56

const normalize = (value, min, max) => (value - min) /( (max - min) || 1)
function makeContext(products, users) {
    const age = users.map(u => u.age)
    const price = products.map(p => p.price)

    const minAge = Math.min(...age)
    const maxAge = Math.max(...age)
    const minPrice = Math.min(...price)
    const maxPrice = Math.max(...price)
    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index ]
        }))
    
    const categoryIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index ]
        }))
    
    //computar a média de didade dos compradores por produto
    //(Ajuda a personalizar as recomendações)
    
    const midAge = minAge + (maxAge - minAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })
    
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
            ageSums[product.name] / ageCounts[product.name] :
            midAge
           
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    return {
        products,
        users,
        colorsIndex,
        categoryIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        //dimensões do vetor de entrada para o modelo: idade, preço, categoria, cor
        dimensions: 2 + categories.length + colors.length,
    }
}

const oneHotWeighted = (index, length, weight) => 
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    return tf.tidy(() => {
        const price = tf.tensor1d([
            normalize(
                product.price,
                context.minPrice,
                context.maxPrice
            ) * WEIGHTS.price
        ])

        const age = tf.tensor1d([
            (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
        ])

        const category = oneHotWeighted(
            context.categoryIndex[product.category],
            context.numCategories,
            WEIGHTS.category
        )
        const color = oneHotWeighted(
            context.colorsIndex[product.color],
            context.numColors,
            WEIGHTS.color
        )

        return tf.concat1d(
            [price, age, category, color]
        )
    })
}


function encoderUser(user, context) {
    if(user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
        .mean(0)
        .reshape([
            1, context.dimensions
        ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), //preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), //categoria é ignorada
            tf.zeros([context.numColors]) //cor é ignorada
        ]
    ).reshape([1, context.dimensions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
    .filter(user => user.purchases.length)
    .forEach(user => {
        const userVector = encoderUser(user, context).dataSync()
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync()

            const label = user.purchases.some(purchase => purchase.name === product.name ? 1 : 0)
            //Combinar user + product
            inputs.push([...userVector, ...productVector])
            labels.push(label)
        
            })
            })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimensions: context.dimensions * 2,
        // Tamanho = userVector + productVector
    }
}



async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimensions],
            units: 128,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )
    model.add(
            tf.layers.dense({
                units: 32,
                activation: 'relu'
            })
        )

        model.add(
            tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            })
        )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                    });
            }
        }
    })
    
    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(products, users);
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

     _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
  
}
function recommend(user, ctx) {
    if(!_model) return;
    const context = _globalCtx
    const userVector = encoderUser(user, context).dataSync()
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    });
    const inputTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputTensor)
    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index]
        }
        
})
    
    const sortedItems = recommendations
    .sort((a, b) => b.score - a.score)

    postMessage({
        type: workerEvents.recommend,
         user,
         recommendations: sortedItems
    });
    
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
