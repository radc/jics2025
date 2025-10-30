import requests

# Use o novo token que você gerar no BotFather
TOKEN = "8176289669:AAFpf0QPNU3LNWThSdX6vGMS4UdKcSIrVfw"
CHAT_ID = 1463494848
MESSAGE = "Olá! Essa é uma mensagem enviada pelo meu bot no Telegram."

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

payload = {
    "chat_id": CHAT_ID,
    "text": MESSAGE
}

response = requests.post(url, data=payload)

if response.status_code == 200:
    print("Mensagem enviada com sucesso!")
else:
    print(f"Erro {response.status_code}: {response.text}")
