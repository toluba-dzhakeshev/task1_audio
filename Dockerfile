# Используем официальный образ Python (например, 3.10-slim)
FROM python:3.10-slim

# Устанавливаем зависимости системы (например, ffmpeg для pydub)
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем весь код проекта в контейнер
COPY . .

# Запускаем бота
CMD ["python", "bot.py"]
