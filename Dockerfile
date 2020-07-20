FROM rocm/tensorflow:rocm3.3-tf1.15-dev

RUN apt-get update && \
        apt-get install -y \
        git \
        wget \
        unzip \
        vim \
        zip \
        curl \
        yasm \
        pkg-config \
        nano \
        tzdata \
        libgtk2.0-dev \
        graphviz && \
    rm -rf /var/cache/apk/*

RUN pip3 install matplotlib pandas seaborn

RUN pip3 install keras==2.3.1

WORKDIR /workspace
