# hash:sha256:0614079a017f408c7199d4dee32f297768a38f8ab961e7f7ab2c1e601218d92e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN mamba install -y --channel=conda-forge \
        dask==2024.9.0 \
        dask-image==2024.5.3 \
        distributed==2024.9.0 \
        jupyterlab==3.6.2 \
        matplotlib==3.9.2 \
        numpy==2.1.1 \
        pandas==2.2.3 \
        python==3.11.10 \
        s3fs==2024.9.0 \
        scikit-image==0.24.0 \
        scipy==1.14.1 \
        seaborn==0.13.2 \
        zarr==2.18.3 \
        s5cmd==2.2.2 \
    && mamba clean -ya

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
