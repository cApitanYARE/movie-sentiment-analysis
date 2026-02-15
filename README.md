<h1>Sentiment analysis</h1>
I decided to find out more about neural networks and train a model for sentiment analysis based on IMDB reviews.

For this task, I chose a Sequential neural network, which works well for this type of problem. 

This description will consist of three points:
1. Preparing data for training
2. Training the model
3. HTML interface and using FastAPI

## 1. Preparing data for training 
First of all, we need to clean the dataset by removing stop words, numbers, HTML symbols, URLs (https…), and a few other elements that may interfere with accurate predictions. 

For this, we used the libraries `stopwords` and `word_tokenize`.

Next need to tokenize, lemmatization and normalization text. 
For this, we used the libraries `WordNetLemmatizer`, `autocorrect `.

More details about data preparation can be found in the file **sentimentNoteBook.ipynb**.

## 2. Training the model
After the initial data cleaning, we need to transform the text data into numerical vectors suitable for training. For this, we used the libraries `Tokenizer` and `pad_sequences`.

Next, we need to split the data into training and testing sets. For this, we used the library function `train_test_split`.

Almost the final step of this stage is to choose the model architecture, compile it, define the loss function, and set a few other parameters. For this, we used`tensorflow` and `keras`.

The final step is to train the model using the `model.fit` function, where we need to specify the training data, number of epochs, batch size, validation data, and, of course, the metrics to evaluate the model’s performance.

More details about data preparation can be found in the file **sentimentNoteBook.ipynb**.

## 3. HTML interface and using FastAPI
I decided to create a visual interface using an HTML file and use FastAPI to interact with the model for analyzing text.
![Screenshot](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/6ede4353bfa83a223a3bda4e1b4d24e30c10eea5/img/web.png)

Below, you can see the result after entering text and clicking the "Check Mood" button.
![Screenshot](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/6ede4353bfa83a223a3bda4e1b4d24e30c10eea5/img/p.jpg)
![Screenshot](https://github.com/cApitanYARE/movie-sentiment-analysis/blob/6ede4353bfa83a223a3bda4e1b4d24e30c10eea5/img/n.jpg)

**Soon, I’ll try to host my HTML file and FastAPI application.** 
