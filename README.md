Install Python 3.7

```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
wget https://www.python.org/ftp/python/3.7.6/Python-3.7.6.tgz
tar -xf Python-3.7.6.tgz
cd Python-3.7.6
./configure --enable-optimizations
make -j 8
sudo make altinstall
python3.7 --version
```

```bash
virtualenv -p python3.7 env
source env/bin/activate 
pip install -U -r requirements.txt 
```

```bash
python mainCartPole.py
```
