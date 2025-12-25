import requests

# ================= PASTE YOUR KEYS HERE =================
TELEGRAM_TOKEN = "8457525565:AAGRA4pJ64LKXh0gSWBf9p6sYnIpxqjkSYY"  # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = "7780448652"      # e.g. "987654321"
# ========================================================

def test_message():
    print(f"Testing Telegram Bot...")
    print(f"Token: {TELEGRAM_TOKEN[:10]}... (hidden)")
    print(f"Chat ID: {TELEGRAM_CHAT_ID}")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': "🚨 Test Alert: If you see this, your bot is working!"}
    
    try:
        response = requests.post(url, data=data)
        print(f"\nResponse Code: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 200:
            print("\n✅ SUCCESS! Check your phone.")
        else:
            print("\n❌ FAILED. Check the error message above.")
            
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to Telegram. {e}")

if __name__ == "__main__":
    test_message()