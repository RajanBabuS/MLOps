import pandas as pd
from sklearn.model_selection import train_test_split

#loading data directly from the registered folder only
df = pd.read_csv("data/tourism.csv")#loading data directly from the registered folder only

#CustomerID: This is a unique random identifier per customer. 
#It provides no predictive power and causes overfitting if kept.
df.drop(columns=["CustomerID"], inplace=True)

#setting up X and y variables

X = df.drop(columns=["ProdTaken"])
#ProdTaken: This is your target/label column (the value your model wants to predict). 
#It must be separated from your features 
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
