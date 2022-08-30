from gtts import gTTS
import os
import playsound
tts=gTTS(text="Hello World",lang="en")
filename = "test.mp3"
tts.save(filename)
playsound.playsound(filename)
os.remove(filename)



def speak(text):
    tts = gTTS(text=text, lang='en')

    filename = "abc.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
