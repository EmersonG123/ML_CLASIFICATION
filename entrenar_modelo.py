import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

# Definición de modelos
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Crear directorio para modelos si no existe
if not os.path.exists('modelos'):
    os.makedirs('modelos')

# Cargar datos de spam
df = pd.read_csv("env/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'mensaje']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Vectorizar
vectorizador = CountVectorizer(stop_words='english')
X = vectorizador.fit_transform(df['mensaje'])
y = df['label']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Guardar vectorizador
joblib.dump(vectorizador, 'modelos/vectorizador.pkl')

# Definición de modelos a entrenar
modelos = {
    "MultinomialNB": MultinomialNB(),
    "SVC": SVC(kernel='sigmoid', gamma=1.0, probability=True),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(solver='liblinear', penalty='l1'),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=50, random_state=2)
}

# Entrenar y evaluar cada modelo
resultados = []
for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    
    # Entrenamiento
    modelo.fit(X_train, y_train)
    
    # Evaluación
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Guardar modelo
    joblib.dump(modelo, f'modelos/{nombre}.pkl')
    
    # Guardar resultados
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy,
        'Reporte': report
    })
    
    print(f"{nombre} entrenado y guardado. Accuracy: {accuracy:.2f}")

# Guardar resumen de resultados
resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv('modelos/resumen_modelos.csv', index=False)

print("\nEntrenamiento completado. Resumen guardado en 'modelos/resumen_modelos.csv'")