import whisper
import google.generativeai as genai
# Load the model (smallest one for speed)
model = whisper.load_model("tiny")
question = model.transcribe("audiotest.mp3") 
genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
model = genai.GenerativeModel("gemini-1.5-flash")

response_text = ""
li = []
final_list=[]

def get_user_input(statement):
    """ Get sentiment classification from Gemini API """
    global response_text
    question = f"Classify as Positive, Negative, or Neutral sentiment. Give answer in one word: '{statement}'"
    print(question)  # Debug print
    response = model.generate_content(question)
    response_text = response.text.strip()  # Clean response

def separator():
    """ Split text into sentences based on periods """
    global li, question
    li = [sentence.strip() for sentence in question.split('.') if sentence.strip()]  # Remove empty parts

def printer():
    separator()  # Split into sentences
    for sentence in li:
        get_user_input(sentence)  # Get sentiment
        final_list.append((sentence,response_text))
        print(f"Sentence: {sentence}\nSentiment: {response_text}\n")  # Print result
        
if __name__ == '__main__':
    printer()