# FakeNewsDetector
Verifies the given news title on web and classifies if it is real or fake 

Steps to run the app
Clone the repo:

git clone <https://github.com/rnbabu/FakeNewsDetector>
cd your-project

Install dependencies:

pip install -r requirements.txt

get tavily API key from  https://app.tavily.com/ and past it in .env file

Change the model info as required in fact_check_agent.py

Run the app:

streamlit run fact_check_agent.py
