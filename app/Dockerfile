
FROM python:3

EXPOSE 5000

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt
# # RUN pip3 install numpy scipy
RUN pip3 install eventlet


COPY . /

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]