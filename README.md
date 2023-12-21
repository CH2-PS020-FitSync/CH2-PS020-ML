# FitSync's Machine Learning
Machine Learning part of FitSync's Workout Recommender and Nutrition Recommender

# Table of Contents
* [Machine Learning Team](#-machine-learning-team)
* [Installation](#%EF%B8%8F-installation)
* [Usage](#-usage)
* [Models Documentation](#-models-documentation)
* [API Documentation](#-api-documentation)

# <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" alt="flask" width="30" height="30"/> Machine Learning Team
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Bangkit-ID</th>
            <th>Socials</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Darrel Cyril Gunawan</td>
            <td>M108BSY1617</td>
            <td style="text-align: center;">
                <a href="https://github.com/Darrelcyril29/">
                    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Github" />
                </a>
                <a href="https://www.linkedin.com/in/darrel-cyril-85517ba1/">
                    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
                </a>
            </td>
        </tr>
        <tr>
            <td>Nigel Kusdenata</td>
            <td>M108BSY1102</td>
            <td style="text-align: center;">
                <a href="https://github.com/NigelKus/">
                    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Github" />
                </a>
                <a href="https://www.linkedin.com/in/nigel-kusdenata-32910528b/">
                    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
                    </a>
            </td>
        </tr>
        <tr>
            <td>Steven Tribethran</td>
            <td>M694BSY0582</td>
            <td style="text-align: center;">
                <a href="https://github.com/Insisted/">
                    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Github" />
                </a>
                <a href="https://www.linkedin.com/in/steven-tribethran/">
                    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
                </a>
            </td>
        </tr>
    </tbody>
</table>

# ⚙️ Installation
1. Clone the repository: 
    ```bash
    git clone https://github.com/CH2-PS020-FitSync/CH2-PS020-ML.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CH2-PS020-ML
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

# 💼 Usage
1. Set up the environments variable:
    ```properties
    FLASK_RUN_HOST=<HOST>
    FLASK_RUN_PORT=<PORT>
    FITSYNC_CONN=<DB_CONNECTION>
    FITSYNC_USER=<DB_USER>
    FITSYNC_PASS=<DB_PASS>
    ```
2. Run the application:
    ```cmd
    flask run [--host=<HOST>] [--port=<PORT>]
    ```

# 📝 Models Documentation
We made two models and serve it on Flask API, those are:
* [Workout Recommender](#-workout-recommender)
* [Nutrition Recommender](#-nutrition-recommender)

## 💪 Workout Recommender
<p align="center">
    <img src="https://msha096.github.io/blog/assets/img/movie_dataset.png"/>
</p>

Our workout recommendation system is inspired by a [TensorFlow](https://www.tensorflow.org/) implementation of [LightFM](https://arxiv.org/abs/1507.08439) on [this article](https://towardsdatascience.com/a-performant-recommender-system-without-cold-start-problem-69bf2f0f0b9b), [LightFM introduction](https://msha096.github.io/blog/lightfm/). We modifies the implementation to align with our specific objectives, then simplify its architecture but also improve its performance on our dataset. The reason we adopt LightFM is based on its ability to mitigate the effect of [Cold start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)) problem in a recommendation, We start building the model by creating a [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) for collaborative filtering and generate a user-item embeddings then calculate the dot product to capture the relation between features and characteristics, enabling  the recommendation inference without relying on historical interaction data.

<p align="center">
    <img src="model\embedding_workout.png"/>
</p>

Our model's performance is evaluated using the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) metric. To optimize the training process, we chose the [Adam](https://golden.com/wiki/Adam_(support_vector_machine)) optimizer. Adam has the benefit of adaptive learning rate techniques, the main reason why we chose Adam compared to traditional [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).<br/><br/>

**Final metrics**:
```
MSE: 2.0721
VAL_MSE: 2.2145
MAE: 1.2287
VAL_MAE: 1.2760
```
<p align="center">
    <img src="model\workout_embedding_error.png"/>
</p>

## 🥗 Nutrition Recommender
Our nutrition recommendation system employs a straightforward neural network architecture consisting of two computational layers: an input layer and a hidden layer. The input layer receives the raw features, while the hidden layer applies [ReLU (Rectified Linear Unit)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation functions, introducing non-linearity to capture complex patterns in the data. The output layer, utilizing a linear activation function, is designed for regression-based tasks. This architecture is particularly suited for predicting continuous values, making it well-suited for regression applications in our recommendation system.

<p align="center">
    <img src="model\nutrition_reg.png"/>
</p>

Our model's performance is evaluated using the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) metric. To optimize the training process, we chose the [Adam](https://golden.com/wiki/Adam_(support_vector_machine)) optimizer. Adam has the benefit of adaptive learning rate techniques, the main reason why we chose Adam compared to traditional [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).<br/><br/>

**Final metrics**:
```
MSE: 6498.6909
VAL_MSE: 6597.4810
MAE: 40.2510
VAL_MAE: 40.1201
```

<p align="center">
    <img src="model\nutrition_error.png"/>
</p>

# <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/flask/flask-original.svg" alt="flask" width="30" height="30"/> API Documentation
The API contains two kinds of endpoint for inference
1. Workout Prediction ➤ [`/workout_predict`](#-workout_predict-post)
2. nutrition Prediction ➤ [`/nutrition_predict`](#-nutrition_predict-post)

<br/>**Base URL**:<br/>
Development: https://fitsync-ml-api-k3bfbgtn5q-et.a.run.app/ <br/>
Production: https://prod-fitsync-ml-api-k3bfbgtn5q-et.a.run.app/

## 🔗 /workout_predict `POST`
**Body:**
* `UserId`: STRING -🔸Required
<br/><br/>

**Successful Responses:**<br/>
🟢 **200 OK**

```json
{
    "data": [
        "<id_string>",
        ...
    ],
    "status": {
        "code": 200,
        "message": "Success"
    }
}
```

## 🔗 /nutrition_predict `POST`
**Body:**
* `UserId`: STRING -🔸Required
<br/><br/>

**Successful Responses:**<br/>
🟢 **200 OK**

```json
{
    "data": {
        "Estimated_Calories": "<float>",
        "Estimated_Carbohydrates": "<float>",
        "Estimated_Fat": "<float>",
        "Estimated_Protein_Mean": "<float>"
    },
    "status": {
        "code": 200,
        "message": "Success"
    }
}
```

