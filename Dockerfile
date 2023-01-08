# Must use a Cuda version 11+
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y python3 python3-pip git
RUN apt-get install -y wget
RUN apt-get install -y build-essential python3-dev

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN=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

ARG KEY
ARG SECRET
ENV KEY=$KEY
ENV SECRET=$SECRET

ENV LD_LIBRARY_PATH=/opt/conda/lib



RUN wget https://huggingface.co/spaablauw/FloralMarble/resolve/main/FloralMarble-400.pt
RUN wget https://huggingface.co/spaablauw/FloralMarble/resolve/main/FloralMarble-250.pt
RUN wget https://huggingface.co/spaablauw/FloralMarble/resolve/main/FloralMarble-150.pt
RUN wget https://huggingface.co/spaablauw/PhotoHelper/resolve/main/PhotoHelper.pt
RUN wget https://huggingface.co/tolerantpancake/LysergianDreams/resolve/main/LysergianDreams-3600.pt
RUN wget https://huggingface.co/spaablauw/UrbanJungle/resolve/main/UrbanJungle.pt
RUN wget https://huggingface.co/spaablauw/CinemaHelper/resolve/main/CinemaHelper.pt
RUN wget https://huggingface.co/tolerantpancake/NegativeMutation/resolve/main/NegMutation-2400.pt
RUN wget https://huggingface.co/spaablauw/CarHelper/resolve/main/CarHelper.pt
RUN wget https://huggingface.co/spaablauw/HyperFluid/resolve/main/HyperFluid.pt
RUN wget https://huggingface.co/joachimsallstrom/Double-Exposure-Embedding/resolve/main/dblx.pt
RUN wget https://huggingface.co/GeneralAwareness/Ppgra/resolve/main/ppgra.pt
RUN wget https://huggingface.co/Conflictx/VikingPunk/resolve/main/VikingPunk.pt
RUN wget https://huggingface.co/spaablauw/GigaChad/resolve/main/GigaChad.pt
RUN wget https://huggingface.co/ProGamerGov/knollingcase-embeddings-sd-v2-0/resolve/main/kc16-v4-5000.pt
RUN wget https://huggingface.co/spaablauw/ActionHelper/resolve/main/ActionHelper.pt

RUN wget -q --no-cache --no-dns-cache https://github.com/pkurzend/stable-diffusion-scripts/raw/master/stableDiffusionDataset.py
RUN wget -q --no-cache --no-dns-cache https://github.com/pkurzend/stable-diffusion-scripts/raw/master/CLIPTokenizerWithEmbeddings.py
RUN wget -q --no-cache --no-dns-cache https://github.com/pkurzend/stable-diffusion-scripts/raw/master/textual_inversion_dreambooth.py

RUN wget -q --no-cache --no-dns-cache https://github.com/pkurzend/stable-diffusion-scripts/raw/master/train_dreambooth21.py
RUN wget -q --no-cache --no-dns-cache https://github.com/pkurzend/stable-diffusion-scripts/raw/master/train_dreambooth.py


# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py




