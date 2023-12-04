# [Project 2. End-to-end Machine Learning HIV Classifier Project with Deployment on Render Using Flask](https://github.com/normatovbekzod/hiv_classifier/tree/main)
In this project, we want to tackle the lack of awareness of the importance of regular testing for HIV especially among the at-risk groups. The main problem that workers at the Republican Centre for the Fight Against AIDS (RCFAA, for short) observed is that most HIV-positive patients only find out about their diagnosis after developing the first symptoms. In the case of at-risk groups, this increases the chance of them passing the virus to another person through sexual intercourse or repeated use of syringes. Hence, there is a lack of preventive awareness of the importance of regular testing especially when the person observes certain behaviors or lifestyles associated with a higher risk of HIV infection. 

Deploying a real-time HIV test predictor can if not solve then at least address all these issues. As such, the goal of this project is to build an HIV test result classifier based on the HIV data from the Republican Centre for the Fight Against AIDS in Tashkent, Uzbekistan, and then deploy it so that the predictor can easily be accessed through a QR code. The project consists of two stages:
The goal of the classifier is to encourage people who fall into at-risk groups including sex workers and injective drug users to get tested as well as to increase the amount of people getting tested at the annual free HIV testing day. 

### The choice of Render 
Render is a unified cloud to build and run apps and websites with free SSL, global CDN, private networks, and automatic deploys from Git. Compared to platforms like Heroku, it offers a Free tier for all web services, which makes it the best candidate to host our Flask app. It is relatively straightforward to deploy a Flask app on Render given only a few requirements and its compatibility with Git will allow us to deploy our app seamlessly. 
Below is an overview of the framework of this project.
<p align="center">
  <img width="625" height="400" src="image/framework.png">
</p>

## Stage 1. Training an HIV test result classifier

We have preprocessed HIV data coming from the RCFAA itself that includes 9978 test results of the people tested at the Centre. Our aim is to clean the data, transform it, and train a model using cross-validation as well as do a grid search for each model type including Logistic Regression, Decision Tree, and Random Forest. It is of utmost importance to use cross-validation while optimizing for the best hyperparameters in order not to avoid overfitting. We then compare the best model of each type by plotting ROC curves and comparing AUC scores for each model. It is important to understand that we aim to get as high of a recall as possible because our model should be able to predict HIV-positive test results for as many actual positive test results in the dataset as possible. However, at the same time, we do not want to compromise the performance of the model by prioritizing recall alone, which might result in too many false positives. Hence, while our aim is to get the highest recall possible, we also should aim for a high AUC score in order to maintain excellent model performance. So, we aim to have a good balance between a high AUC score and a high recall. We do that by optimizing for both recall score and roc auc score during grid search and then comparing the best models. 

## Stage 2. Deploying the model on Render using Flask app

The output is a web app in the form of a survey with 14 questions. By providing necessary inputs, i.e. answers to the questions, the user can get a real-time prediction of their risk of being HIV positive by clicking the "Predict" button. The app can be accessed through [here](https://hiv-test-survey.onrender.com/) Note: since the end of the project we downgraded to a free tier on Render, in which web services automatically spun down after 15 minutes of inactivity. When you click on the link, Render spins it up again so it can process the request and this can cause a response delay of up to 30 seconds.

After the user inputs answers to 14 questions asked of them, the webpage then sends these inputs to the Flask app, which transforms them into a valid input for the classifier. The classifier then spits out a prediction, which is then displayed on a new web page for the user to see and take further action. There is also a picture that indicates where and when the user can get tested for HIV for free.
