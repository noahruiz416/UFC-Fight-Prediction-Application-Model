# UFC Fight Prediction Application User Guide 

## Summary 
This file contains information regarding the UFC prediction application and how fight fans can utilize the model to make predictions on upcoming UFC fights. In addition keep in mind that the example you will see initially is just how the model would run given no input data.

### Input Data 
In order to actually utilize the fight prediction tool, you will need to enter the information prompted by the boxes on the left. If you do not fill out all the boxes the model will not work and no prediction will be made. 
After that as long as the information you entered is correct you should get a basic prompt back with probabilities assigned to which fighter is most likely to win. Below that prompt you will also see a graph that explains 
which variables led to that prediction, which I will explain how to interpret below.

### Prediction Interpretation 
Too keep this descirption as non technical as possible we can think of each dot corresponding to how much a given feature affects the prediction of our model. 
For example if we see that B_age is to the far right, we can interpret that as the age of the blue fighter is pushing the model to make its prediction. In addition if that 
dot were to the far left we can interpret that as the age of the blue fighter, is pushing the model to predict that the blue fighter will lose the fight.

### Explanation of Values Inputted
The actual parameters that we put into the model are quite simple and are as follows: 
  - Blue Corner Reach in Centimeters
  - Red Corner Reach in Centimeters
  - Blue Corners Age
  - Red Corners Age
  - Blue Corners Name
  - Red Corners Name
  - Red Corner Losses
  - Blue Corner Losses

With each corner reffering to the corner that the given fighter is in.
  

### Precautions when using the tool
I would heavily recommend that the model be used on fighters who have more than 5 fights in the UFC. This is because the model was trained on a wide variety of fighters, and in order to make predictions based on "Fighter Name", 
the model will need to have seen that given fighters name multiple times in order to make accurate predictions. Because of this you should avoid putting new or unknown fighters or fights into this predictive model, as it will 
only the other features to make predictions. (ie fighter physical attributes and record) However it is possible to not consider the name of the fighter in the model, in order to do so just input "Blue" into the blue corner name and "Red" into the red corner name, this will lead to the model only taking into consideration the other variables in the model.
