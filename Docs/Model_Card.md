# Model card for UFC Fight Prediction Model

Sections and prompts from the [model cards paper](https://arxiv.org/abs/1810.03993), v2.

Jump to section:

- [Model details](#model-details)
- [Intended use](#intended-use)
- [Factors](#factors)
- [Metrics](#metrics)
- [Evaluation data](#evaluation-data)
- [Training data](#training-data)
- [Quantitative analyses](#quantitative-analyses)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Model details

- Person or organization developing model: Noah Ruiz, Student at ASU, Mathematics (Statistics)

- Model date: 
  08/25/22
  
- Model version: 
  - 1.0
 
- Final Model type: 
  - Catboost Classification

- Information about training algorithms, parameters, fairness constraints or other applied
  approaches, and features:   
  - Model takes in around 8 parameters in order to make predictions of which corner will win a fight
  - Parameters include: "Red Losses", "Blue Losses", "Red Age", "Blue Age", "Red Reach", "Blue Reach", "Red Name", "Blue Name"
  - Parameters were chosen through a combination of domain expertise and various recursive feature selection techniques/shapley values 
 
- Paper or other resource for more information:
  - Model: https://catboost.ai/en/docs/
  - Shapley Values: https://shap.readthedocs.io/en/latest/
  - WebScraper Repo: https://github.com/WarrierRajeev/UFC-Predictions

- Where to send questions or comments about the model: 
  - noahruiz416@gmail.com

## Intended use

_Use cases that were envisioned during development._

Review section 4.2 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Primary intended uses
- Allow fight fans to predict UFC fights ideally for fighters that have more than 5 fights in the UFC
- Aid in the understanding of the features that lead to a fighter winning 

### Primary intended users
- UFC Fight Fans  

### Out-of-scope use cases
- Using the model to make bets on fights (due to the nature of fightining, there is still noise our model does not account for)
- Predicting fights for fighters who we have very little data on (less than 5 fights in UFC)
- Predicting fights outcomes for fighters not in the UFC

## Factors

### Relevant factors
- The model was developed on fight data after 2001, which is when the Unified Rules of MMA were created.
- Training data ranges from 1993 - Aug 2022

## Evalutation Metrics
- F1 Score
- Accuracy 

### Model performance measures
- For this project we chose F1 score as the main metric as we want a balance between both precision and recall 
- In addition since the class balance is not too severe accuracy also works as a secondary metric for the model

### Approaches to uncertainty and variability
- Since the final model only takes into account 8 parameters, there is a very high chance that the model does not properly map a population prediction function 
- In addition fighting is an extremley voilitaile sport and there are other factors that the model cannot include such as: fighter mental state and more 

## Training Data

### Datasets
- 'https://github.com/noahruiz416/UFC-Fight-Prediction-Application-Model/tree/main/Data'

### Motivation
- Allow Fight fans to predict UFC fights in a simple web application 

### Preprocessing
- Replacing NA values with 0's. This is a key assumption in our model. However I checked that the distribution of values stays the same even after replacing null values with 0's.
- In addition we only consider values after 2001 when the Unified Rules of MMA were created
- SMOTE was used to fix the slight class imbalance in the training data between red and blue corners winning

## Quantitative analyses

### Unitary results:
- Our final implementation using Catboost recieved the following scores based on our evaluation metrics on test data: 
  - Weighted F1 Score: .69
  - Accuracy: .71

## Ethical considerations

### Data
- Colelcted data could have potential issues if the webscraper does not capture recent fights 
- In addition model drift is a very important issue so I plan on retraining and testing the model at the end of the year 

### Risks and harms
- The model may have an inherent bias towards fighters that are younger and have a longer reach, this circumvents fighters that have a good amoount of experience who still continue to win 

## Caveats and recommendations
- Moving forward I recommend that the model be tested on more out of sample data to see if performance still persists
- In addition I recommend that better SWE practices be implemented and test cases should be implemented to see what the web application is not able to handle.


