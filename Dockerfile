ARG BASE_IMG

FROM ${BASE_IMG}

# Install curl (needed to fetch miniforge installer)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniforge from GitHub
RUN curl -L  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# Set up miniforge environment
ENV PATH=/opt/conda/bin:$PATH
ENV HOME=/home/

RUN <<EOF
# ensure conda environment is always activated
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo ". /opt/conda/etc/profile.d/conda.sh; conda activate base" >> /etc/skel/.bashrc
echo ". /opt/conda/etc/profile.d/conda.sh; conda activate base" >> ~/.bashrc
EOF

RUN cat <<EOF > /opt/conda/.condarc
envs_dirs:
  - /home/.conda/envs
pkgs_dirs:
  - /home/.conda/pkgs
ssl_verify: False
channel_alias: https://anaconda.mgb.org/
channel_priority: strict
channels:
  - conda-forge
EOF

# Copy the environment YAML file into the container
COPY environments/pharos_environment.yaml /tmp/env.yaml

ARG CUDA_VERSION=12.8

# Install dependencies from the YAML file using mamba
RUN CONDA_OVERRIDE_CUDA=$CUDA_VERSION conda env create -n pharos -f /tmp/env.yaml && \
    conda clean --all --yes && \
    echo ". /opt/conda/etc/profile.d/conda.sh; conda activate pharos" >> ~/.bashrc

COPY src /home/src
COPY demos /home/demos

WORKDIR /home
