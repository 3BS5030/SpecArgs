import os
import io
import subprocess
from gtts import gTTS

DATASET_SIZE = 55
OUTPUT_DIR = "./data"
SAMPLE_RATE = 16000

SENTENCES = [
    "The cat sat on the mat",
    "I like to read books",
    "She went to the store",
    "The sun is shining bright",
    "He runs very fast",
    "We are going to the park",
    "The dog barked at the mailman",
    "It is a beautiful day",
    "Please open the door",
    "Can you help me please",
    "The bird flew over the house",
    "My favorite color is blue",
    "They are playing soccer",
    "The food tastes delicious",
    "Good morning how are you",
    "I need a glass of water",
    "The music is very loud",
    "She loves to dance",
    "He drove his car to work",
    "The flowers are blooming",
    "Turn off the light please",
    "I am learning to code",
    "The movie was really good",
    "Let us go for a walk",
    "The baby is sleeping quietly",
    "Please pass the salt",
    "The train arrives at noon",
    "I brush my teeth every day",
    "The sky is very clear tonight",
    "He wrote a letter to his friend",
    "The coffee is too hot",
    "She wears a red dress",
    "The students are studying hard",
    "I live in a small town",
    "The river flows through the valley",
    "Please sit down and relax",
    "The cake tastes very sweet",
    "He plays the guitar well",
    "We visited the museum yesterday",
    "The phone is ringing loudly",
    "I will meet you at noon",
    "The tree has green leaves",
    "She speaks three languages fluently",
    "The rain stopped in the evening",
    "He fixed the broken chair",
    "The children are laughing happily",
    "I enjoy swimming in the pool",
    "The bridge connects two cities",
    "Please write your name here",
    "The pizza is ready to eat",
    "She painted a lovely picture",
    "He reads the newspaper every morning",
    "The store closes at nine o clock",
    "I put on my warm jacket",
    "The cat climbed up the tree",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, sentence in enumerate(SENTENCES):
    idx = f"sample{i+1:03d}"
    wav_path = os.path.join(OUTPUT_DIR, f"{idx}.wav")
    txt_path = os.path.join(OUTPUT_DIR, f"{idx}.txt")

    print(f"[{i+1}/{DATASET_SIZE}] Generating: {sentence}")

    tts = gTTS(text=sentence, lang="en", slow=False)
    mp3_path = os.path.join(OUTPUT_DIR, f"{idx}.mp3")
    tts.save(mp3_path)

    subprocess.run(
        [
            "/tmp/ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffmpeg",
            "-y", "-loglevel", "error",
            "-i", mp3_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            wav_path,
        ],
        check=True,
    )
    os.remove(mp3_path)

    with open(txt_path, "w") as f:
        f.write(sentence + "\n")

print(f"\nDone — {DATASET_SIZE} samples generated in {OUTPUT_DIR}/")
