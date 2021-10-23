# in case of runing with local conda instead of docker

export GOOGLE_APPLICATION_CREDENTIALS=/data/resources/cloud/gcp/keys/eastonlab-b37c04a8f1a5.json
export MLFLOW_TRACKING_URI=http://mlflow_server:5000

python run.py