import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    "Machine_ID": np.arange(1, 101),
    "Temperature": np.random.randint(50, 100, 100),
    "Run_Time": np.random.randint(100, 500, 100),
    "Downtime_Flag": np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)
df.to_csv("sample_data.csv", index=False)




import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(file_path):
    df = pd.read_csv(file_path)
    X = df[['Temperature', 'Run_Time']]
    y = df['Downtime_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model.score(X_test, y_test),accuracy
