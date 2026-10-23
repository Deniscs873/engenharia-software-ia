import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

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
function makeContext(catalog, users) {
    const age = users.map(u => u.age)
    const price = catalog.map(p => p.price)

    const minAge = Math.min(...age)
    const maxAge = Math.max(...age)
    const minPrice = Math.min(...price)
    const maxPrice = Math.max(...price)
    const colors = [...new Set(catalog.map(p => p.color))]
    const categories = [...new Set(catalog.map(p => p.category))]

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
        catalog.map(product => {
            const avg = ageCounts[product.name] ?
            ageSums[product.name] / ageCounts[product.name] :
            midAge
           
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    return {
        catalog,
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

function encondeProduct(product, context) {
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
            context.numCategories,
            WEIGHTS.category
        )

        return tf.concat1d(
            [price, age, category, color]
        )
    })
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json();

    const context = makeContext(catalog, users);
    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encondeProduct(product, context).dataSync()
        }
    })

    debugger
    
    _globalCtx = context
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
