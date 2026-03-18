import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
 
FEATURES = ['LOC','McCabe_CC','n_methods','n_attributes','WMC','DIT','NOC',
            'CBO','RFC','LCOM','n_imports','n_comments','avg_method_len']
 
def compute_code_metrics(code_str):
    return {
        'LOC': len(code_str.split('
')),
        'McCabe_CC': code_str.count('if') + code_str.count('for') +
                     code_str.count('while') + code_str.count('switch') + 1,
        'n_methods': code_str.count('def ') + code_str.count('public ') +
                     code_str.count('private '),
        'n_attributes': code_str.count('self.') + code_str.count('private int') +
                        code_str.count('private String'),
        'avg_method_len': max(1, len(code_str.split('
')) //
                              max(1, code_str.count('def '))),
    }
 
def simulate_dataset(n=500):
    np.random.seed(42)
    X = np.random.randn(n, len(FEATURES))
    X[:, 0] = np.abs(X[:, 0])*100 + 50   # LOC > 0
    X[:, 1] = np.abs(X[:, 1])*5 + 1      # Cyclomatic > 1
    y = (0.3*X[:,0]/100 + 0.4*X[:,1]/10 + 0.2*X[:,7]/5 +
         np.random.randn(n)*0.3 > 1.0).astype(int)
    return X, y
 
X, y = simulate_dataset()
scaler = StandardScaler(); X_s = scaler.fit_transform(X)
models = [('RandomForest', RandomForestClassifier(100, random_state=42)),
          ('GradientBoosting', GradientBoostingClassifier(50, random_state=42)),
          ('LogisticRegression', LogisticRegression(max_iter=500))]
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Defect prediction results (5-fold CV):")
for name, model in models:
    scores = cross_val_score(model, X_s, y, cv=cv, scoring='f1')
    print(f"  {name:20s}: F1={scores.mean():.3f} ± {scores.std():.3f}")
 
best_model = RandomForestClassifier(100, random_state=42).fit(X_s, y)
importances = dict(zip(FEATURES, best_model.feature_importances_))
top = sorted(importances.items(), key=lambda x:-x[1])[:5]
print("\nTop defect predictors:", [f"{k}({v:.3f})" for k,v in top])
