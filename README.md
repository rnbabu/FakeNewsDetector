# FakeNewsDetector
Verifies the given news title on web and classifies if it is real or fake 

- Steps to run the app
#Clone the repo:

- git clone <https://github.com/rnbabu/FakeNewsDetector>

cd your-project

- Create a virtual environment (optional but recommended):
   ```bash
   python -m venv my_env
   ```

- Activate the virtual environment:
   ```bash
   # On Windows
   .\my_env\Scripts\Activate.ps1
   # On macOS and Linux
   source my_env/bin/activate
   ```

- Install project dependencies:
   ```bash
   pip install -r requirements.txt

- Install dependencies:

pip install -r requirements.txt

get tavily API key from  https://app.tavily.com/ and past it in .env file

- if you want to use local model download Ollama and pull the required models like llama3.2:3b, llama3.1:8b..etc, the code is Ollama ready

- Change the model info as required in fact_check_agent.py

Run the app:

streamlit run fact_check_agent.py
