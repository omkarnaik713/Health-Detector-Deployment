FROM python 3.7
COPY .
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python app.py
