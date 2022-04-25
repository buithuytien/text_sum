# from streamlit_summarizer import *
import streamlit as st
import requests


def main():

    
    
    st.title("Abstractive Text/Article Summarizer")
    st.subheader("Summary with NLP")
    raw_text = st.text_area(label = "enter text here", value = "main representative body british jews called wigan chairman dave whelan comments outrageous labelled apology halfhearted whelan set face football association charge responded controversy wigan appointment malky mackay manager telling guardian think jewish people chase money everybody else wigan owner since apologised offence caused facing critical situation club one latics shirt sponsors kitchen firm premier range announced breaking ties club due whelan appointment mackay subject fa investigation sending allegedly racist text messages iain moody former head recruitment cardiff dave whelan left jewish body outraged following comments aftermath malky mackay hiring board deputies british jews vicepresident jonathan arkush said statement dave whelan comments jews outrageous offensive bring club game disrepute halfhearted apology go far enough insult whole group people say would never insult hope ok need see proper apology full recognition offence caused whelan role chair football club responsibility set tone players supporters mackay appointed wigan boss week despite text email scandal racism antisemitism prevail pitch acceptable unchallenged boardroom taking matter football association kick", )
    summary_choice = st.selectbox("Summary Choice",["N-GRAM", "RNN-RNN", "BERT-RNN", "BERT-BERT"])
    post_req = None
    if st.button("Generate Summary"):
        print("Button Pressed")
        if summary_choice == "RNN-RNN":
            pass_url = 'http://c63c-155-33-134-23.ngrok.io/test/' + raw_text
    
        if summary_choice == "BERT-RNN":
            pass_url = 'http://be6e-34-82-13-204.ngrok.io/test/' + raw_text
        
        if summary_choice == "BERT-BERT":
            pass_url = 'http://7a09-34-83-99-59.ngrok.io/test/' + raw_text
        
        if summary_choice == "N-GRAM":
            pass_url = 'http://79b5-155-33-134-23.ngrok.io/test/' + raw_text
        
        post_req = requests.get(url = pass_url)
        st.write(post_req.text)

if __name__ == '__main__':
    main()
