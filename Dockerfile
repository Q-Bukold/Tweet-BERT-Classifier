FROM jupyter/scipy-notebook

RUN pip install jupyter_contrib_nbextensions jupyterlab-lsp 
RUN pip install 'python-lsp-server[all]'

RUN pip install regex numpy
RUN pip install scikit-learn
RUN pip install pip install transformers[tf-cpu]==4.31.0

#RUN pip install transformers==4.24.0
#RUN pip install simpletransformers==0.63.11
#RUN pip install torch



# on host machine:

## create image:
## -> $docker build -t {IMAGENAME} .
## -> The . at the end of the docker build command tells that Docker should look for the Dockerfile in the current directory.

## create container:
## $docker run -p 8888:8888 -v {host_dir}:/home/jovyan {IMAGENAME}
## fast: $docker run -p 8888:8888 -v $(pwd):/home/jovyan {IMAGENAME}
## -> this Folder should be the host_dir (e.g. /Users/qbukold/Heavy_Projects/GitHub/yt_schnulzen)

## Use container:
## Paste 2. URL given by Notebook afer docker run into Websoftware like Firefox
## -> The URL should be printed under "Or copy and paste one of these URLs:" in your Terminal
