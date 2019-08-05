
# FROM python:3.7.3-alpine
# WORKDIR /app
# ENV FLASK_APP app.py
# ENV FLASK_RUN_HOST 0.0.0.0
# # RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev
# # RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# RUN apk add --no-cache gcc musl-dev linux-headers
# COPY requirements.txt requirements.txt
# # RUN pip install perl
# # FROM alpine:3.7.3
# # RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/main" > /etc/apk/repositories
# # RUN echo "http://dl-cdn.alpinelinux.org/alpine/latest-stable/community" >> /etc/apk/repositories
# # RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev
# # RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
# # RUN pip install numpy scipy pandas matplotlib
# # install openblas
# # RUN  echo -e '[DEFAULT]\n\
# # library_dirs = /usr/lib/openblas/lib\n\
# # include_dirs = /usr/lib/openblas/lib\n\n\
# # [atlas]\n\
# # atlas_libs = openblas\n\
# # libraries = openblas\n\n\
# # [openblas]\n\
# # libraries = openblas\n\
# # library_dirs = /usr/lib/openblas/lib\n\
# # include_dirs = /usr/lib/openblas/lib'  >> site.cfg 
# # RUN wget https://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz \
# # 	&& tar -xf v0.3.6.tar.gz \
# # 	&& cd OpenBLAS-0.3.6/ \
# # 	&& make BINARY=64 FC=$(which gfortran) USE_THREAD=1 \
# # 	&& make PREFIX=/usr/lib/openblas install
# # RUN ATLAS=/usr/lib/openblas/lib/libopenblas.so LAPACK=/usr/lib/openblas/lib/libopenblas.so pip install scipy==1.3

# # RUN pip install --no-cache-dir numpy scipy pandas 
# #!/bin/sh
# # # RUN  ./run.sh
# # ADD run.sh /
# # RUN chmod +x /run.sh
# RUN pip install -r requirements.txt
# # COPY . .
# # CMD ["python", "app.py"]
# EXPOSE 5000
# ENTRYPOINT [ "python" ]
# FROM python:latest
# WORKDIR /app
# ENV FLASK_APP app.py
# ENV FLASK_RUN_HOST 0.0.0.0
# RUN pip install --upgrade pip
# RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt
# RUN pip3 install numpy scipy
# CMD ["python","app.py"]


# CMD [ "app.py" ]

# Dockerfile-flask
# We simply inherit the Python 3 image. This image does
# not particularly care what OS runs underneath
FROM python:3
# Set an environment variable with the directory
# where we'll be running the app
# ENV APP /app
# # Create the directory and instruct Docker to operate
# # from there from now on
# RUN mkdir $APP
# WORKDIR $APP
# # Expose the port uWSGI will listen on
EXPOSE 5000
# # Copy the requirements file in order to install
# # Python dependencies
# COPY requirements.txt .

# Install Python dependencies
# RUN pip3 install -r requirements.txt

# We copy the rest of the codebase into the image
# COPY . .
# Finally, we run uWSGI with the ini file we
# created earlier
# CMD [ "uwsgi", "--ini", "app.ini" ]
# CMD ["python","app.py"]


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