#Rabiralsya s problemoi zapuska na mac
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import tempfile
import torch
import whisper
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# -------------------------
# Загрузка моделей
# -------------------------
print("Загружаем модель Whisper...")
whisper_model = whisper.load_model("base")  # можно заменить на "large"

print("Загружаем модель анализа настроения...")
model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

# Метки для сентимент-анализа ("Neutral" не используется)
labels = ["Negative", "Neutral", "Positive"]

# -------------------------
# Функция для предварительной обработки аудио (MP3 → WAV + шумоподавление)
# -------------------------
def preprocess_audio(input_mp3: str, output_wav: str):
    # Конвертация MP3 → WAV (моно, 16 кГц)
    audio = AudioSegment.from_mp3(input_mp3)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_wav, format="wav")

    # Уменьшение шума
    y, sr = librosa.load(output_wav, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_wav, reduced_noise, sr)

# -------------------------
# Хэндлер для команды /start
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Пришли мне MP3-файл, и я выполню транскрипцию и анализ настроения."
    )

# -------------------------
# Хэндлер для приёма аудиофайлов (MP3)
# -------------------------
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #Определяем, был ли файл отправлен как документ или как аудио
    file_id = None
    file_name = None
    if update.message.document:
        file_id = update.message.document.file_id
        file_name = update.message.document.file_name
    elif update.message.audio:
        file_id = update.message.audio.file_id
        file_name = update.message.audio.file_name
    else:
        await update.message.reply_text("Пожалуйста, отправьте MP3-файл.")
        return

    #Проверяем, что файл имеет расширение .mp3
    if not file_name.lower().endswith(".mp3"):
        await update.message.reply_text("Файл не является MP3.")
        return

    await update.message.reply_text("Обработка файла... Это может занять некоторое время.")

    #Создаём временные файлы для исходного MP3 и результирующего WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        mp3_path = tmp_mp3.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name

    try:
        #Скачиваем MP3-файл с серверов Telegram
        file = await context.bot.get_file(file_id)
        await file.download_to_drive(mp3_path)

        #Предварительная обработка: конвертация и шумоподавление
        preprocess_audio(mp3_path, wav_path)

        #Транскрибируем аудио с помощью Whisper
        result = whisper_model.transcribe(
            wav_path,
            language="ru",
            condition_on_previous_text=False
        )
        transcription_text = result["text"]

        #Анализ настроения полученной транскрипции
        inputs = tokenizer(
            transcription_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        sentiment_idx = torch.argmax(scores).item()
        sentiment_label = labels[sentiment_idx]
        sentiment_score = scores[sentiment_idx].item() * 100

        #В примере: если не Positive – считаем Negative
        if sentiment_label == "Positive":
            deal_status = f"Positive ({sentiment_score:.2f}%)"
        else:
            deal_status = f"Negative ({sentiment_score:.2f}%)"

        #Формируем ответ для пользователя
        reply_text = (
            f"Транскрипция:\n{transcription_text}\n\n"
            f"Анализ настроения: {deal_status}"
        )
        await update.message.reply_text(reply_text)

    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при обработке файла:\n{e}")

    finally:
        # Удаляем временные файлы
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

# -------------------------
# Основная функция запуска бота
# -------------------------
if __name__ == '__main__':
    bot_token = "7428135947:AAEOcgyj5k5mcHomY7Lk27tgk9ETb3nSe3s"

    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.AUDIO, handle_audio))

    print("Бот запущен. Ожидаю сообщений...")
    app.run_polling()
