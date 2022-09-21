# in case of runing with local conda instead of docker

export GOOGLE_APPLICATION_CREDENTIALS=/data/resources/cloud/gcp/keys/eastonlab-b37c04a8f1a5.json
export MLFLOW_TRACKING_URI=http://192.168.1.154:5001

python3 run.py.

nohup python /path/to/test.py > output.log &
