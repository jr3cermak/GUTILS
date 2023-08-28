FROM phusion/baseimage:focal-1.2.0

LABEL org.opencontainers.image.description="GUTILS as a docker image."
LABEL org.opencontainers.image.authors="Kyle Wilcox <kyle@axiomdatascience.com>"
LABEL org.opencontainers.image.url="https://git.axiom/SECOORA/GUTILS/"
LABEL org.opencontainers.image.source="https://git.axiom/SECOORA/GUTILS/"
LABEL org.opencontainers.image.licenses="MIT"


# Use baseimage-docker's init system
CMD ["/sbin/my_init"]
ENV KILL_PROCESS_TIMEOUT 30
ENV KILL_ALL_PROCESSES_TIMEOUT 30

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
        binutils \
        build-essential \
        bzip2 \
        ca-certificates \
        file \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pwgen \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup CONDA (https://hub.docker.com/r/continuumio/miniconda3/~/dockerfile/)
ENV MAMBAFORGE_VERSION 23.1.0-1
ENV MAMBAFORGE_DOWNLOAD Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh
RUN curl -k -L -O "https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/${MAMBAFORGE_DOWNLOAD}" && \
    curl -k -L -O "https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/${MAMBAFORGE_DOWNLOAD}.sha256" && \
    sha256sum --check "${MAMBAFORGE_DOWNLOAD}.sha256" && \
    /bin/bash ${MAMBAFORGE_DOWNLOAD} -b -p /opt/conda && \
    rm ${MAMBAFORGE_DOWNLOAD} && \
    /opt/conda/bin/mamba clean -afy

ENV PATH /opt/conda/bin:$PATH

COPY environment.yml /tmp/environment.yml
RUN mamba env update \
        -n base \
        -f /tmp/environment.yml \
        && \
    mamba clean -afy

COPY pip-requirements.txt /tmp/pip-requirements.txt
RUN pip install \
        --no-deps \
        --force-reinstall \
        --ignore-installed \
        -r /tmp/pip-requirements.txt

ENV PATH /opt/conda/bin:$PATH

ENV GUTILS_DEPLOYMENTS_DIRECTORY /gutils/deployments
ENV GUTILS_ERDDAP_CONTENT_PATH /gutils/erddap/content
ENV GUTILS_ERDDAP_FLAG_PATH /gutils/erddap/flag
VOLUME ["${GUTILS_DEPLOYMENTS_DIRECTORY}", "${GUTILS_ERDDAP_CONTENT_PATH}", "${GUTILS_ERDDAP_FLAG_PATH}"]

RUN mkdir -p /etc/my_init.d && \
    mkdir -p /gutils
COPY docker/init/* /etc/my_init.d/

ENV GUTILS_VERSION 3.2.0

ENV PROJECT_ROOT /code
RUN mkdir -p "$PROJECT_ROOT"
COPY . $PROJECT_ROOT
RUN cd $PROJECT_ROOT && pip install --no-deps .
WORKDIR $PROJECT_ROOT
