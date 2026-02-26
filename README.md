# Movie Sentiment Analysis

A neural network trained on the IMDB dataset (available at [Kaggle]https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). 

The project consists of the following parts:
1. **Model**
2. **Backend**
3. **Frontend**
4. **Deploy**

## 1. Model

&nbsp;&nbsp;&nbsp;&nbsp;You can find the model's description in the `notebook/sentimentNoteBook.ipynb` file.

## 2. Backend

&nbsp;&nbsp;&nbsp;&nbsp;The backend uses the following libraries: `FastAPI`, `pydantic`, `uvicorn`, `dockerfile`, `docker-compose`, and TensorFlow Lite (code located in fastAPI/api.py).   
&nbsp;&nbsp;&nbsp;&nbsp;First, the model is loaded and prepared for analysis. `TensorFlow Lite` and `NLTK` are used to perform these tasks.  
&nbsp;&nbsp;&nbsp;&nbsp;I use `TensorFlow` Lite instead of `TensorFlow` to reduce RAM usage. After training and testing the model in the `notebook/sentimentNoteBook.ipynb` file, the model is saved as a `.keras` file. To further reduce memory usage, it is converted from `.keras` to `.tflite`. The converter can be found in `fastAPI/convert.py`.    
&nbsp;&nbsp;&nbsp;&nbsp;To learn more about the backend API methods, visit `https://movie-sentiment-analysis-yfg1.onrender.com/docs`.

&nbsp;&nbsp;&nbsp;&nbsp;To reduce the Docker image size, a multi-stage build is used, which is defined in `fastAPI/Dockerfile`.

## 3. Frontend

&nbsp;&nbsp;&nbsp;&nbsp;The review submission page is shown below:

![img/web.png](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/23da45822fc0c4c9e9ae6effe390dc2197fd03e9/img/web.png)

&nbsp;&nbsp;&nbsp;&nbsp;After writing some text, you should press the **"Check Mood"** button. A JavaScript script using `fetch` will then send a `"POST"` request to the API from **point 2 (Backend)**. If the request is successful, the frontend will receive a response indicating the sentiment of the text: either `positive` or `negative`.  
&nbsp;&nbsp;&nbsp;&nbsp;Some examples are shown below:
![img/p.jpg](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/23da45822fc0c4c9e9ae6effe390dc2197fd03e9/img/p.jpg)
![img/n.jpg](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/23da45822fc0c4c9e9ae6effe390dc2197fd03e9/img/n.jpg)

## 4. Deploy

For deploying the project, I used a free instance on [Render](https://render.com/) with a 512 MB memory limit.

- **Frontend:** [https://movie-sentiment-analysis-f2as.onrender.com](https://movie-sentiment-analysis-f2as.onrender.com)  
- **API (Backend):** [https://movie-sentiment-analysis-yfg1.onrender.com](https://movie-sentiment-analysis-yfg1.onrender.com)  
