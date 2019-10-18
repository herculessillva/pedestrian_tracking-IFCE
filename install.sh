PATH_REPO=$PWD
virtualenv .env -p python3 --no-site-packages
chmod 777 $PATH_REPO/.env/*
source $PATH_REPO/.env/bin/activate
pip install -r requirements.txt
cd ..
git clone https://github.com/thtrieu/darkflow.git
cd darkflow/
python3 setup.py build_ext --inplace
pip install .
cd $PATH_REPO
deactivate
