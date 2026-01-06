import joblib 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.ensemble import RandomForestClassifier

def main(): 
    # 1) Load dataset 
    iris = load_iris() 
    X = iris.data 
    y = iris.target 
    # 2) Split 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
    ) 
    # 3) Train model 
    model = RandomForestClassifier(random_state=42) 
    model.fit(X_train, y_train) 
    # 4) Evaluate 
    y_pred = model.predict(X_test) 
    acc = accuracy_score(y_test, y_pred) 
    print("Accuracy:", acc) 
    print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
    # 5) Save model 
    joblib.dump(model, "iris_model.pkl") 
    print("\nModel saved as iris_model.pkl") 
if __name__ == "__main__": 
    main() 
