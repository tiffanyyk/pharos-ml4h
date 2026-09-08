# PHAROS (Pulmonary Hypertension Assessment fROm ECG Signals)

## PHAROS Demo
Two sample aligned waveform segments are provided from the publicly available [MIMIC-IV Waveform Database](https://physionet.org/content/mimic4wdb/0.1.0/) as an example of how to run our trained weights on real data. We also provide a dummy data batch generator to demonstrate how to structure custom data into a format our model expects. The demo script, model weights, and config files are found in [`demos/pharos_multiclass_demo/`](demos/pharos_multiclass_demo/)

## PHAROS Model Code
Source code for PHAROS and contrastive pre-training are found in [`src/models`](src/models). Our datasets cannot be released due to IRB restrictions.

## Environment Setup
Dependencies can be found in [`environments/pharos_environment.yaml`](environments/pharos_environment.yaml), or you can use the [`build_image.sh`](build_image.sh) script to create the docker image for running the code.
