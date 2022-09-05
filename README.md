# UFC-Fight-Predction-Application-Model Project Directory
<img width="500" alt="Screen Shot 2022-08-23 at 9 06 30 AM" src="https://user-images.githubusercontent.com/88412646/186207419-047394a0-7d1a-487b-b9ed-238c0357f82a.png">

# Status Update:
V.1 of the project is Complete! I deployed the model to a simple heroku server and leveraged mercury in order to take advatnage of a simple jupyter notebook framework. As of now I am in the process of writing a user manual/guide and documentation. In addition V.1.1 will come out relatively soon and I plan on adding a few more features on the back and front end of this project. 

# Summary: 
V.1 of this project incorporates a simple user interface and a machine learning classification model, to allow fight fans to predict UFC fights, given simple user inputted fight statistics. The model was trained over an iterative process in which multiple modeling techniques and feature combinations were considered until we got the best out of sample results. Overall the model itself reports accuracy of 68% with an F1 Score (Our Metric of Choice) of 70% on the out of sample data. The model used is 'CatBoost' an ensemble learning method that allows for categorical variables as input features, this allowed us to consider fighter name in the model. In addition I used shapley values to find the features that account for the marginal contribution to the predicted values.

In doing so I found these features to add the most to the models classifications: 
 - Fighter Age 
 - Fighter Reach 
 - Fighter Name 
 - Fighter Losses

After building the model I leveraged mercury to host a jupyter notebook in an interactive user interface, that allows users to input values and make predictions with the trained CatBoost model. Finally I deployed the model to a Heroku web server which hosts the application for anyone to use.

# Project Link: 
- https://noahruiz-about-me.herokuapp.com/app/1

## Resource Guide:
Within this readme you will find various links, which lead to different files in this repository. 

### Docs/User Manual: 
Contains simple read me files that explain various aspects of the model.
- https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/blob/main/Docs/Model_Card.md

### Notebooks
Notebooks for this project, go here if you want a nice interactive format for this project additionally I recommend using google collab for uploading:
- https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/tree/main/Notebooks

### Prototype
Initial prototypes of the dataset, baseline models and other Ad-hoc tests:
- https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/tree/main/Prototypes

### Data:
Dataset used for this analysis, the data was collected with an open source webscraper that chekcs the UFC stats website for fight data:
- https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/tree/main/Data

### Models:
Final Models for this project that will be used to predict fights:
- https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/tree/main/Models

References: 
- https://github.com/WarrierRajeev/UFC-Predictions (Rajiev Warrier, for open source UFC webscraper)
- https://catboost.ai (for modeling technology)
- https://shap.readthedocs.io (for model explainability and interpretation)
