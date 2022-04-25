import torch
from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from rnn_summarizer_github import EncoderRNN, DecoderRNN, prod_evaluate
decoderpath = 'rnn_models/decoder.pt'
encoderpath = 'rnn_models/encoder.pt'


    # encoder = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8').to(device)



encoder1 = torch.load(encoderpath,map_location=torch.device('cpu'))
decoder1 = torch.load(decoderpath,map_location=torch.device('cpu'))


# !pip install fastapi nest-asyncio pyngrok uvicorn
app = FastAPI()
@app.get('/test/{raw_text}')
async def home(raw_text = "main representative body british jews called wigan chairman dave whelan comments outrageous labelled apology halfhearted whelan set face football association charge responded controversy wigan appointment malky mackay manager telling guardian think jewish people chase money everybody else wigan owner since apologised offence caused facing critical situation club one latics shirt sponsors kitchen firm premier range announced breaking ties club due whelan appointment mackay subject fa investigation sending allegedly racist text messages iain moody former head recruitment cardiff dave whelan left jewish body outraged following comments aftermath malky mackay hiring board deputies british jews vicepresident jonathan arkush said statement dave whelan comments jews outrageous offensive bring club game disrepute halfhearted apology go far enough insult whole group people say would never insult hope ok need see proper apology full recognition offence caused whelan role chair football club responsibility set tone players supporters mackay appointed wigan boss week despite text email scandal racism antisemitism prevail pitch acceptable unchallenged boardroom taking matter football association kick"):
        summary_result = prod_evaluate(raw_text, encoder1, decoder1)
        return summary_result
ngrok_tunnel = ngrok.connect(8001)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8001)










