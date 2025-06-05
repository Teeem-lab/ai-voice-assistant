from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pinecone import Pinecone
import openai
import requests
import os


PINECONE_API_KEY = 'pcsk_...'
PINECONE_INDEX_NAME = 'japanese-lessons'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")


pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -- FastAPI instance --
app = FastAPI()

# -- Optionally, initialize chat memory --
conversation_memory = [
    {"role": "system", "content": "You are a helpful Japanese teacher. Use clear and simple Japanese and English, and always reference relevant lesson info if provided."}
]  # <-- This closes the list!

# Set this to True for lesson explanations, or False for natural chat
lesson_mode = False  # Change to False for natural conversation mode

def search_pinecone(query_text, top_k=1):
    embed = client.embeddings.create(input=[query_text], model="text-embedding-3-small").data[0].embedding
    results = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
    if results.matches:
        return results.matches[0].metadata
    else:
        return {}

@app.post("/chat/")
async def chat(audio: UploadFile = File(...)):
    # 1. Save and transcribe audio
    audio_bytes = await audio.read()
    with open("input.mp3", "wb") as f:
        f.write(audio_bytes)
    with open("input.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_text = transcript.text
    print("User said:", user_text)

    # 2. Retrieve relevant lesson content from Pinecone
    lesson_metadata = search_pinecone(user_text)
    print("Pinecone match:", lesson_metadata)

    # 3. Auto-detect if the user is asking for a lesson/explanation
    lower_text = user_text.lower()
    needs_lesson = any(word in lower_text for word in [
        "how do i", "what does", "explain", "lesson", "mean", "grammar", "say", "translate",
        "in lesson", "mnemonic", "remember", "remind me"
    ])

    if needs_lesson and lesson_metadata:
        lesson_info = (
            f"\nLesson info:\n"
            f"Japanese: {lesson_metadata.get('jp_text', '')}\n"
            f"English: {lesson_metadata.get('en_translation', '')}\n"
            f"Grammar: {lesson_metadata.get('grammar_point', '')}\n"
            f"Image: {lesson_metadata.get('image_description', '')}"
        )
        prompt = (
            f"User said: {user_text}\n"
            f"{lesson_info}\n"
            f"Please answer as a kind Japanese teacher, referencing the lesson information to help the student understand and practice."
        )
    else:
        prompt = (
            f"User said: {user_text}\n"
            f"Please have a natural, friendly Japanese conversation without explicit lesson or grammar explanations. Respond as a native speaker."
        )

    conversation_memory.append({"role": "user", "content": prompt})

    # 4. Get GPT-4o reply
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_memory
    )
    reply_text = response.choices[0].message.content
    print("Assistant reply:", reply_text)
    conversation_memory.append({"role": "assistant", "content": reply_text})

    # 5. Convert reply to speech with ElevenLabs
    tts_res = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        json={"text": reply_text}
    )
    with open("reply.mp3", "wb") as f:
        f.write(tts_res.content)
    # 6. Return the audio file
    return FileResponse("reply.mp3", media_type="audio/mpeg")


