### Diabetes Detection Application backend

### Flask app

from flask import Flask, render_template, request

app = Flask(__name__)



# Loading pickle Model
import pickle as pk
pickle_file = open("My_final_diabetes_RF_model.pickle","rb")
model = pk.load(pickle_file)

# importing all neccessary things for Model-prediction
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("diabetes_prediction_dataset.csv")


from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range=(0,1))

num_cols = ['age','bmi','HbA1c_level','blood_glucose_level']
sc.fit(df[num_cols])


# func to get all preprocessing done for the input record-data
def preprocess(age,bmi,HbA1c_level,blood_glucose_level,gender,smoking_history):
    scaled = sc.transform([[age, bmi, HbA1c_level, blood_glucose_level]])           # this returns a 2D-array of scaled values
    # convert gender into Numericals
    if gender=='Female':
        gender1 = [1,0,0]
    elif gender=='Male':
        gender1 = [0,1,0]
    elif gender=='Other':
        gender1 = [0,0,1]
    # convert smoking_history into Numericals
    if smoking_history=='no_info':
        smoking_history1 = [1,0,0,0,0,0]
    elif smoking_history=='current':
        smoking_history1 = [0,1,0,0,0,0]
    elif smoking_history=='ever':
        smoking_history1 = [0,0,1,0,0,0]
    elif smoking_history=='former':
        smoking_history1 = [0,0,0,1,0,0]
    elif smoking_history=='never':
        smoking_history1 = [0,0,0,0,1,0]
    elif smoking_history=='not_current':
        smoking_history1 = [0,0,0,0,0,1]
    # joining all the preprocessed feature_values into list
    features = []
    for i in scaled[0]:
        features.append(i)
    features.extend(gender1)
    features.extend(smoking_history1)
    # Converting features-list into array
    features_dict = {"age":[features[0]], "bmi":[features[1]], "HbA1c_level":[features[2]], "blood_glucose_level":[features[3]],
                     "gender_Female":[features[4]], "gender_Male":[features[5]], "gender_Other":[features[6]], "smoking_history_No Info":[features[7]],
                     "smoking_history_current":[features[8]], "smoking_history_ever":[features[8]], "smoking_history_former":[features[9]], 
                     "smoking_history_never":[features[10]], "smoking_history_not current":[features[11]]}

    features_df = pd.DataFrame(features_dict)
    return features_df
    

# _______________________________________________________________________________________________




@app.route('/')
def home():
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def detect():
    if request.method=='POST':
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['glucose_level'])
        gender = str(request.form['gender'])
        smoking_history = str(request.form['smoking']) 
    input = preprocess(age,bmi,HbA1c_level,blood_glucose_level,gender,smoking_history)

    output = model.predict(input)

    if output==1:
        pred = "Diabetic."
    elif output==0:
        pred = "not Diabetic."

    return render_template("index.html", prediction="Person is {}".format(pred))





if __name__ == "__main__":
    app.run(debug=True) 









