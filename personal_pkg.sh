#!/bin/bash
cd /src
python -m pip install --prefix=$(python -m site --user-base) -e /src/datajoint-python/
python -m pip install --prefix=$(python -m site --user-base) -e /src/attorch/
python -m pip install --prefix=$(python -m site --user-base) -e /src/neuro_data/
python -m pip install --prefix=$(python -m site --user-base) -e /src/featurevis/
python -m pip install --prefix=$(python -m site --user-base) -e /src/static-networks/
python -m pip install --prefix=$(python -m site --user-base) -e /src/utils/
python -m pip install --prefix=$(python -m site --user-base) -e /src/nnfabrik/
python -m pip install --prefix=$(python -m site --user-base) -e /src/nnsysident/
python -m pip install --prefix=$(python -m site --user-base) -e /src/neuralpredictors/
python -m pip install --prefix=$(python -m site --user-base) -e /src/sensorium/

# python -m pip install --prefix=$(python -m site --user-base) -e 'neuralpredictors~=0.0.1'
# python -m pip install --prefix=$(python -m site --user-base) -e /src/insilico-stimuli/
# python -m pip install --prefix=$(python -m site --user-base) -e /src/controversial-stimuli/
jupyter lab --ip=0.0.0.0 --port=9999 --allow-root --NotebookApp.token=$JUPYTER_PASSWORD --no-browser
