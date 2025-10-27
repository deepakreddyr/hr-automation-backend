import asyncio
import websockets
import json
import base64
# from divya_prompt import parse_candidate_response
from openai import OpenAI
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)

async def handle_stream(websocket):
    async for message in websocket:
        data = json.loads(message)
        if data.get("event") == "media":
            audio_b64 = data["media"]["payload"]
            audio_bytes = base64.b64decode(audio_b64)

            # 1️⃣ Send audio to OpenAI Realtime for transcription
            transcript = await openai.realtime.transcribe(audio_bytes)

            # 2️⃣ Get AI response using Divya prompt
            ai_text = await openai.realtime.generate(transcript, prompt="Divya prompt here")

            # 3️⃣ Convert AI text response to audio and send back
            ai_audio = await openai.realtime.text_to_speech(ai_text)
            await websocket.send(ai_audio)

            # 4️⃣ Extract candidate info
            candidate_info = parse_candidate_response(transcript)
            print("Candidate info:", candidate_info)

        elif data.get("event") == "start":
            print("Stream started")
        elif data.get("event") == "stop":
            print("Stream stopped")

async def main():
    async with websockets.serve(handle_stream, "0.0.0.0", 8765):
        print("WebSocket server listening on port 8765")
        await asyncio.Future()  # run forever

asyncio.run(main())

def parse_candidate_response(transcript: str):
    # Example regex-based parsing; you can improve with OpenAI API
    candidate = {}
    if "CTC" in transcript:
        candidate["current_ctc"] = "Extracted from transcript"
    if "expected" in transcript:
        candidate["expected_ctc"] = "Extracted from transcript"
    if "experience" in transcript:
        candidate["experience"] = "Extracted from transcript"
    if "skills" in transcript:
        candidate["skills"] = ["Python", "SQL"]  # example
    return candidate