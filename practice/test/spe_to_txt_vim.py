import speech_recognition as sr
#arecord -D hw:2,0 -c 1 -r 16000 -f S16_LE -d 10 cap3.wav

r = sr.Recognizer()
with sr.AudioFile('cap4.wav') as source:
    #r.adjust_for_ambient_noise(source,duration=5)
    audio_text = r.listen(source)
    #print(type(audio_text))
    #try:
    text = r.recognize_google(audio_text)
    print(text)
     #   print('Converting audio transcripts into text ...')
      #  print(text)
    #except:
     #   print('Sorry.. run again...')
