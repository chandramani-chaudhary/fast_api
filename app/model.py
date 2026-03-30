#Machine learning Model
import pandas as pd
import joblib #to convert model file to binary file(pkl file)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/HR_Dataset_Refresh.csv')
MODEL_PATH = '/Users/chandramani/Desktop/fastapi/model/logistic_empstatus.pkl'
SCALER_PATH = '/Users/chandramani/Desktop/fastapi/model/logistic_scalar.pkl'
def predict_logostic():
    features =['PerfScoreID','Salary','PositionID','EngagementSurvey', 'EmpSatisfaction','SpecialProjectsCount', 'DaysLateLast30','Absences']
    target='Termd'
    x=df[features]
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,  stratify=y
    )   
    scaler=StandardScaler()
    x_train_scale=scaler.fit_transform(x_train) #xi=mean/sd
    x_test_scale = scaler.transform(x_test)
    model =LogisticRegression(
        solver='liblinear',class_weight ='balanced',random_state=42
    )
    model.fit(x_train_scale,y_train)
    model.predict(x_test_scale)

    joblib.dump(model,MODEL_PATH)
    joblib.dump(scaler,SCALER_PATH)
    return model, scaler
# load  model
def load_model_scaler():
    model =joblib.load(MODEL_PATH)
    scaler =joblib.load(SCALER_PATH)
    return model , scaler
    