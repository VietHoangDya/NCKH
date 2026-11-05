import pyttsx3

giong_doc = pyttsx3.init()

for i, giong in enumerate(giong_doc.getProperty("voices")):
    print("So:", i)
    print("Ten:", giong.name)
    print("ID:", giong.id)
    print("------------------")
