from pyngrok import ngrok

# Open tunnel
public_url = ngrok.connect(8501)
print("Public URL:", public_url)

# Keep running
input("Press Enter to stop...")