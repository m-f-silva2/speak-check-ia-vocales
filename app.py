from core import convert_audio_to_spectrograms, PhonemeRecognitionService
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import os
import soundfile as sf

app = Flask(__name__)
CORS(app, resources={r"/*": {"Access-Control-Allow-Origins": "*"}})
model = PhonemeRecognitionService()

@app.route("/api/", methods=["POST"])
def most_frequent_phoneme():
    # guarda el audio
    recording = request.files["recording"]
    # obtener espectogramas
    spectrograms = convert_audio_to_spectrograms(recording)
    # llamar model, hacer predicciones
    preds = model.predict(spectrograms)

    # filtrar la clase ruido
    phonemes = list(filter(lambda pred: pred["class"] != "noise", preds))
    preds = phonemes if len(phonemes) > 0 else preds
    phoneme = max(preds, key=lambda x: x["percentage"])
    print("fonema:",phoneme)
    return jsonify(
        {
            "pronunciation": "correct",
            "phoneme": phoneme,
        }
    )
def process_audio(file_path):
    # Cargar el audio a su frecuencia original (48,000 Hz)
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Resamplear a 16,000 Hz
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    return audio, sample_rate

@app.route("/api/word/<pattern>", methods=["POST"])
def validate_phoneme_pattern(pattern: str):   
    recording = request.files['recording']
    # Guardar temporalmente el archivo para inspeccionarlo
    temp_file_path = "test_recording.wav"
    recording.save(temp_file_path)
    processed_audio, sample_rate = process_audio(temp_file_path)

    # Guardar el audio procesado en un nuevo archivo
    processed_file_path = "processed_recording.wav"
    sf.write(processed_file_path, processed_audio, sample_rate)
    print(f"Processed audio saved at: {processed_file_path}")

    try:
        # Procesar el archivo
        spectrograms = convert_audio_to_spectrograms(processed_file_path)
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(processed_file_path):
            #os.remove(processed_file_path)
            print(f"Archivo temporal {processed_file_path} eliminado.")

    predictions = model.predict(spectrograms)
    phoneme = None
    percentage = 0.0
    phonemes = []

    for prediction in predictions:
        if prediction["class"] == phoneme:
            if prediction["percentage"] > percentage:
                percentage = prediction["percentage"]
                phonemes[-1]["percentage"] = prediction["percentage"]
            continue

        phoneme = prediction["class"]
        percentage = prediction["percentage"]
        phonemes.append(prediction)

    
    start_pattern = None
    phonemes = list(filter(lambda pred: pred["class"] != "noise", phonemes))

    for i, phoneme in enumerate(phonemes):
        if phoneme["class"] == pattern[0]:
            start_pattern = i
            break
    print("phonemes",phonemes)
    print("estar",start_pattern," fonema:", pattern)
    print("phoneme:",phoneme)

    if start_pattern == None:
        result = {"word": pattern, "score": 0, "phonemes": []}
        return jsonify(result)

    default_phoneme = {"class": "unknown", "percentage": 0.0}
    predicted = phonemes[start_pattern : start_pattern + len(pattern)]
    predicted = predicted + [default_phoneme] * (len(pattern) - len(predicted))

    total_percentage = sum(phoneme["percentage"] for phoneme in predicted)
    average = total_percentage / len(predicted) if len(predicted) > 0 else 0
    result = {"word": pattern, "score": average, "phonemes": predicted}

    return jsonify(result)
    

@app.route('/')
def home():
    return "Running app"


if __name__ == "__main__":
    # pyinstaller --onefile --console --clean server.py --name "backend"
    spectrograms = convert_audio_to_spectrograms("./models/recording.wav")
    model.predict(spectrograms)
    

    # version app
    #app.run(port=4000, debug=False)

    # version runanyway
    app.run(port=5000, debug=False, host="0.0.0.0")

    # deploy production
    #app.run(debug=False, host="0.0.0.0")
