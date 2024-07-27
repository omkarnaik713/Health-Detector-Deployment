FROM python
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libhdf5-dev
RUN apt-get install -y libsndfile1
RUN apt-get install -y ffmpeg
RUN mkdir -p /app/logs
RUN mkdir -p /app/upload
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python app.py
