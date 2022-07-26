brew install pyenv
pyenv install 3.9.13
pyenv global 3.9.13
pip install virtualenv
virtualenv env
source env/bin/activate
ipython kernel install --name "env" --user
pip install matplotlib seaborn sklearn plotly babyplots xgboost tensorflow-macos