#PYTHON PART
FROM python:3.7

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Source code
COPY . /app


RUN apt-get update && apt-get install -y unzip

RUN unzip /app/sim_app.zip -d /app

RUN ["chmod", "+x", "/app/sim_app/linux.x86_64"]
RUN ["chmod", "+x", "/app/main.py"]

EXPOSE 8090:8080

CMD /app/sim_app/linux.x86_64 & sleep 60 && python /app/main.py
