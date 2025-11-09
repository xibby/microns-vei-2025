FROM atlab/pytorch:v0.0.10-py38-torch1.7.0-cuda11-dj_atlab
RUN apt-get -y update && apt-get  -y install ffmpeg
WORKDIR /src
RUN git clone https://github.com/atlab/attorch.git
RUN python -m pip install imageio ffmpy h5py opencv-python statsmodels
RUN python -m pip install --prefix=$(python -m site --user-base) -e ./attorch/
WORKDIR /notebooks
