import subprocess

VIRGIN_URL = "http://icecast.unitedradio.it/Virgin.mp3"

def play_virgin():
    print("Avvio streaming Virgin Radio con mpg123...")
    try:
        subprocess.run(["mpg123", VIRGIN_URL])
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    play_virgin()
