
module load python3/3.9.10

pip install --upgrade pip

python3 -m venv .

source bin/activate

pip install -r requirements.txt

python3 src/main.py