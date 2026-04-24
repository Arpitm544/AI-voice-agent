import speech_recognition as sr
import os
import sys
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

from agent import process_interaction, client
from rich.console import Console
from rich.panel import Panel

console = Console()

def speak(text: str):
    """Uses macOS built-in say command for Text-to-Speech."""
    console.print(Panel(text, title="🤖 Agent", border_style="cyan"))
    # Escaping quotes to prevent shell injection issues
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    os.system(f'say "{safe_text}"')

def listen_to_user(recognizer, microphone):
    """Listens to the microphone and returns the transcribed text."""
    with microphone as source:
        console.print("[dim yellow]Listening...[/dim yellow]")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None
    
    try:
        console.print("[dim yellow]Recognizing via Groq Whisper...[/dim yellow]")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio.get_wav_data())
            temp_path = temp_audio.name
            
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("audio.wav", file.read())
            )
            
        text = transcription.text.strip()
        os.remove(temp_path)
        
        if text:
            console.print(f"[bold green]👤 You:[/bold green] {text}")
        return text
    except Exception as e:
        console.print(f"[bold red]Transcription error: {e}[/bold red]")
        try:
            os.remove(temp_path)
        except:
            pass
        return None

def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("Initializing Voice AI Agent...")
    speak("Hello, I am ready. How can I help you today?")
    
    # Store conversation history
    messages = []
    
    while True:
        try:
            user_text = listen_to_user(recognizer, microphone)
            if user_text:
                cleaned_text = user_text.lower().strip()
                exit_phrases = ["exit", "quit", "stop", "exist", "goodbye", "bye", "stop now", "can we stop now", "stop the agent"]
                if cleaned_text in exit_phrases:
                    speak("Goodbye!")
                    break
                
                # Prevent Memory Overload (Sliding Context Window)
                # If messages list gets too long (e.g. > 15), keep the system prompt [0] and the last 10 interactions
                if len(messages) > 15:
                    messages = [messages[0]] + messages[-10:]
                
                # Append user message
                messages.append({"role": "user", "content": user_text})
                
                # Process with Grok/Groq safely
                try:
                    messages, response_text = process_interaction(messages)
                    
                    # Speak response
                    if response_text:
                        speak(response_text)
                except Exception as e:
                    console.print(f"[bold red]Error communicating with AI: {e}[/bold red]")
                    speak("I'm sorry, I encountered an error connecting to my brain. Let's try that again.")
                    
        except KeyboardInterrupt:
            speak("Goodbye!")
            break

if __name__ == "__main__":
    main()
