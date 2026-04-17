FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    coreutils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libstdc++6 \
    libgcc-s1 \
    zlib1g \
    bzip2 \
    xz-utils \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/akoya

COPY bin/ /opt/akoya/bin/
COPY conf_files/ /opt/akoya/conf_files/
COPY env_archive/sp_env_packclone.tar.gz /tmp/sp_env_packclone.tar.gz

RUN mkdir -p /opt/conda_env && \
    tar -xzf /tmp/sp_env_packclone.tar.gz -C /opt/conda_env && \
    /opt/conda_env/bin/python /opt/conda_env/bin/conda-unpack && \
    rm -f /tmp/sp_env_packclone.tar.gz

ENV PATH=/opt/conda_env/bin:${PATH}
ENV PYTHONNOUSERSITE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "bin/run_AKOYA.py", "conf_files/conf_BK22-SKI-27-FO-1-S34-D1.yaml"]