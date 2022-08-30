import os
os.system("arecord -D hw:2,0 -c 1 -r 16000 -f S16_LE -d 5 cap3.wav")
import speech_recognition as sr
#arecord -D hw:2,0 -c 1 -r 16000 -f S16_LE -d 10 cap3.wav

r = sr.Recognizer()
with sr.AudioFile('cap3.wav') as source:
    #r.adjust_for_ambient_noise(source,duration=5)
    audio_text = r.listen(source)
    try:
        text = r.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
        print(text)
    except:
        print('Sorry.. run again...')
