# MA Crossover
Download S&P500 stock data and check for MA20/200 crossings from below to above in the last 10 days.   
Then check, which sectors are currently strong and filter the ma crossing hits against the strong sectors.   
   
   

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

# ToDo
- add seasonality check   
- create ma_crossover.ipynb   

# Version
ma_crossover.ipynb: The development version (yet to come)   
ma_crossover.py: command line version