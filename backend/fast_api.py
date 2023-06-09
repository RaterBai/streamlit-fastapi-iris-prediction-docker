from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

target_dict = {"0" : "setosa", 
               "1" : "versicolor", 
               "2" : "virginica"}

class User_input(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

app = FastAPI()
@app.post("/prediction")
def operate(input:User_input):
    input_data = [[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]]
    model = pickle.load(open("./model/iris_model.sav", 'rb'))
    
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0]

    prediction_str = f"The predicted species is **{target_dict[str(prediction)]}**  |"
    prediction_prob_str = f"The probability of being setosa is **{np.round(prediction_prob[0], 3)}**  |The probability of being versicolor is **{np.round(prediction_prob[1], 3)}**  |The probability of being virginica is **{np.round(prediction_prob[2], 3)}**  | "
    prediction_prob_str = prediction_prob_str.replace(target_dict[str(prediction)], f'**{target_dict[str(prediction)]}**')
    return(prediction_str+prediction_prob_str)

if __name__ == '__main__':
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)