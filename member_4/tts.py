import platform
import subprocess


class TextToSpeechError(RuntimeError):
    pass


def speak_text(text):
    text = (text or "").strip()
    if not text:
        raise TextToSpeechError("No transcription text to read.")

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except ImportError:
        pass
    except Exception as exc:
        raise TextToSpeechError(f"pyttsx3 failed: {exc}") from exc

    if platform.system().lower() == "windows":
        escaped = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{escaped}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as exc:
            raise TextToSpeechError(f"Windows speech synthesis failed: {exc}") from exc

    raise TextToSpeechError("Install pyttsx3 to enable text-to-speech on this system.")
