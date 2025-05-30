import joblib
import os
from pre_processamento import limpar_texto

# atenção aos caminhos #
MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_final.pkl")

print("Carregando modelo e vetorizador...")
vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

def classificar_email(texto):
    texto_limpo = limpar_texto(texto)
    vetor = vectorizer.transform([texto_limpo])
    predicao = model.predict(vetor)[0]
    return predicao

# exemplos #
emails = [
    "Congratulations! You've been selected for a $1000 Walmart gift card. Click here to claim now!",
    "Limited time 0ffer! L0w-interest loan approved for you. C@ll now!",
    "Hi John, just confirming our meeting at 10am tomorrow. Let me know if you need anything.",
    "Your invoice for the last project is attached. Please verify and send the signed copy back.",
    "Aumente seu Pênis!"
]

print("\nTestando e-mails de exemplo:\n")
for i, texto in enumerate(emails, 1):
    resultado = classificar_email(texto)
    print(f"{i}. {resultado.upper()} => {texto}\n")

