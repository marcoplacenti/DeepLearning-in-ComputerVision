To run the code on DTU HPC follow this steps:

- connect to HPC
- clone this repository in it and move to the right folder
    ```console
    git clone https://github.com/marcoplacenti/DeepLearning-in-ComputerVision.git
    cd DeepLearning-in-ComputerVision/project-1.1
    ```
- load the python module
    ```console
    module load python3/3.9.10
    ```
- optionally run (recommended)
    ```console
    02514sh
    ```
- upgrade pip 
    ```console
    pip install --upgrade pip
    ```
- create the virtual environment and activate it
    ```console
    python3 -m venv .
    source bin/activate
    ```
- install dependencies
    ```console
    pip install -r requirements.txt
    ```
- run the code
    ```console
    python3 src/main.py
    ```

I will create a bash script that will do all this steps in one go, so we don't have to worry about it anymore.
