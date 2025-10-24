# workshop-2-

## Name : R.Manikandan
## Reg.no: 212223230120
## Binary Classification with Neural Networks on the Census Income Dataset
## Aim:
This project builds a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually based on demographic data from the Census Income dataset.
## Code:
```
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/income (1) (1).csv')

categorical_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
continuous_cols = ['age', 'education-num', 'hours-per-week']
label_col = 'label'

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

cat_data = np.stack([df[col].values for col in categorical_cols], axis=1)
con_data = np.stack([df[col].values for col in continuous_cols], axis=1)
y = torch.tensor(df[label_col].values, dtype=torch.long)

X_cat = torch.tensor(cat_data, dtype=torch.int64)
X_con = torch.tensor(con_data, dtype=torch.float)

X_cat_train, X_cat_test, X_con_train, X_con_test, y_train, y_test = train_test_split(
    X_cat, X_con, y, test_size=0.2, random_state=42
)

class TabularModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, out_sz, p=0.4):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        self.hidden = nn.Linear(sum([nf for ni, nf in emb_sizes]) + n_cont, 50)
        self.bn_hidden = nn.BatchNorm1d(50)
        self.out = nn.Linear(50, out_sz)
        self.dropout = nn.Dropout(p)
    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = F.relu(self.bn_hidden(self.hidden(x)))
        x = self.dropout(x)
        x = self.out(x)
        return x

cat_szs = [len(df[col].unique()) for col in categorical_cols]
emb_szs = [(size, min(50, (size + 1)//2)) for size in cat_szs]

torch.manual_seed(42)
model = TabularModel(emb_szs, len(continuous_cols), 2, p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
for epoch in range(epochs):
    y_pred = model(X_cat_train, X_con_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            val_pred = model(X_cat_test, X_con_test)
            val_loss = criterion(val_pred, y_test)
            acc = (val_pred.argmax(1) == y_test).float().mean()
            print(f'Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}')

def predict_new_input(model, label_encoders):
    model.eval()
    sex = input("Enter sex (Male/Female): ")
    education = input("Enter education (e.g., HS-grad, Masters): ")
    marital_status = input("Enter marital status (e.g., Married, Never-married): ")
    workclass = input("Enter workclass (e.g., Private, Federal-gov): ")
    occupation = input("Enter occupation (e.g., Exec-managerial, Craft-repair): ")
    age = float(input("Enter age: "))
    education_num = float(input("Enter education-num: "))
    hours_per_week = float(input("Enter hours-per-week: "))

    cat_values = [
        label_encoders['sex'].transform([sex])[0],
        label_encoders['education'].transform([education])[0],
        label_encoders['marital-status'].transform([marital_status])[0],
        label_encoders['workclass'].transform([workclass])[0],
        label_encoders['occupation'].transform([occupation])[0]
    ]
    con_values = [age, education_num, hours_per_week]

    x_cat = torch.tensor(np.array(cat_values).reshape(1, -1), dtype=torch.int64)
    x_con = torch.tensor(np.array(con_values).reshape(1, -1), dtype=torch.float)

    with torch.no_grad():
        out = model(x_cat, x_con)
        pred = torch.argmax(out, 1).item()

    if pred == 1:
        print("Predicted Income: >50K")
    else:
        print("Predicted Income: <=50K")

predict_new_input(model, label_encoders)
print("Name: Manikandan R")
print("Register No: 212223230120")

```

## Output
<img width="620" height="310" alt="image" src="https://github.com/user-attachments/assets/bba1dbaf-eca5-49ee-b2c7-1f5c5879f162" />
