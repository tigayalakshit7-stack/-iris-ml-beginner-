import joblib 
def main():
    model = joblib.load("iris_model.pkl")
    # Example input: [sepal_len, sepal_wid, petal_len, petal_wid] 
    sample = [[5.1, 3.5, 1.4, 0.2]] 
    pred = model.predict(sample)[0] 
    names = ["setosa", "versicolor", "virginica"] 
    print("Predicted class:", names[pred])

if __name__ == "__main__": 
    main()
