# FitSync's Machine Learning
Machine Learning part of FitSync's Workout Recommender and Nutrition Recommender

# Table of Contents
* [Machine Learning Team](#-machine-learning-team)
* [Installation](#installation)
* [Usage](#usage)
* [API Documentation](#-api-documentation)

# <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" alt="flask" width="30" height="30"/> Machine Learning Team
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Bangkit-ID</th>
            <th>Github</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Darrel Cyril Gunawan</td>
            <td>M108BSY1617</td>
            <td><a href="#">Github</a></td>
        </tr>
        <tr>
            <td>Nigel Kusdenata</td>
            <td>M108BSY1102</td>
            <td><a href="#">Github</a></td>
        </tr>
        <tr>
            <td>Steven Tribethran</td>
            <td>M694BSY0582</td>
            <td><a href="https://github.com/Insisted">Insisted</a></td>
        </tr>
    </tbody>
</table>

# Installation
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

# Usage
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
    flask run [--host=<HOST>] [--port=<PORt>]
    ```

# <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/flask/flask-original.svg" alt="flask" width="30" height="30"/> API Documentation
The API contains two kinds of endpoint for inference
1. Workout Prediction ➤ [`/workout_predict`](#-workout_predict-post)
2. nutrition Prediction ➤ [`/nutrition_predict`](#-nutrition_predict-post)

<br/>**Base URL**:<br/>
https://fitsync-ml-api-k3bfbgtn5q-et.a.run.app/

## 🔗 /workout_predict `POST`
**Body:**
* `UserId`: STRING -🔸Required
<br/><br/>

**Successful Responses:**<br/>
🟢 **200** OK

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
🟢 **200** OK

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

