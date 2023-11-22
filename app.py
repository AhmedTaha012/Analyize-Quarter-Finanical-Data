import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoModelForSequenceClassification,BertForSequenceClassification
import math
import nltk
import torch
from nltk.corpus import stopwords
import spacy
from spacy import displacy
from word2number import w2n
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import en_core_web_sm
import datetime
nlp = en_core_web_sm.load()
nltk.download('punkt')
nltk.download('stopwords')

similarityModel = SentenceTransformer('BAAI/bge-small-en')
sentiment_model = pipeline("text-classification", model="AhmedTaha012/managersFeedback-V1.0.7")
tokenizerQuarter = AutoTokenizer.from_pretrained('AhmedTaha012/nextQuarter-status-V1.1.9')
modelQuarter = BertForSequenceClassification.from_pretrained('AhmedTaha012/nextQuarter-status-V1.1.9')
tokenizerTopic = AutoTokenizer.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification",use_fast=True,token="")
modelTopic = AutoModelForSequenceClassification.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification",token="")
# torch.compile(modelTopic)
tokenizer = AutoTokenizer.from_pretrained("AhmedTaha012/finance-ner-v0.0.9-finetuned-ner")
model = AutoModelForTokenClassification.from_pretrained("AhmedTaha012/finance-ner-v0.0.9-finetuned-ner")
# torch.compile(model)
# torch.compile(model)
nlpPipe = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False
def disable():
    st.session_state["disabled"] = True

def getSpeakers(data):
    if "Speakers" in data:
        return "\n".join([x for x in data.split("Speakers")[-1].split("\n") if "--" in x])
    elif "Call participants" in data:
        return "\n".join([x for x in data.split("Call participants")[-1].split("\n") if "--" in x])
    elif "Call Participants" in data:
        return "\n".join([x for x in data.split("Call Participants")[-1].split("\n") if "--" in x])
def removeSpeakers(data):
    if "Speakers" in data:
        return data.split("Speakers")[0]
    elif "Call participants" in data:
        return data.split("Call participants")[0]
    elif "Call Participants" in data:
        return data.split("Call Participants")[0]
def getQA(data):
    if "Questions and Answers" in data:    
        return data.split("Questions and Answers")[-1]
    elif  "Questions & Answers" in data:
        return data.split("Questions & Answers")[-1]
    elif "Q&A" in data:
        return data.split("Q&A")[-1]
    else:
        return ""
def removeQA(data):
    if "Questions and Answers" in data:    
        return data.split("Questions and Answers")[0]
    elif  "Questions & Answers" in data:
        return data.split("Questions & Answers")[0]
    elif "Q&A" in data:
        return data.split("Q&A")[0]
    else:
        return ""
def clean_and_preprocess(text):
    text=[x for x in text.split("\n") if len(x)>100]
    l=[]
    for t in text:
        # Convert to lowercase
        t = t.lower()    
        # Tokenize text into words
        words = nltk.word_tokenize(t)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]

        # Join the words back into a cleaned text
        cleaned_text = ' '.join(filtered_words)
        l.append(cleaned_text)
    return "\n".join(l)
def replace_abbreviations(text):
    replacements = {
        'Q1': 'first quarter',
        'Q2': 'second quarter',
        'Q3': 'third quarter',
        'Q4': 'fourth quarter',
        'q1': 'first quarter',
        'q2': 'second quarter',
        'q3': 'third quarter',
        'q4': 'fourth quarter',
        'FY': 'fiscal year',
        'YoY': 'year over year',
        'MoM': 'month over month',
        'EBITDA': 'earnings before interest, taxes, depreciation, and amortization',
        'ROI': 'return on investment',
        'EPS': 'earnings per share',
        'P/E': 'price-to-earnings',
        'DCF': 'discounted cash flow',
        'CAGR': 'compound annual growth rate',
        'GDP': 'gross domestic product',
        'CFO': 'chief financial officer',
        'GAAP': 'generally accepted accounting principles',
        'SEC': 'U.S. Securities and Exchange Commission',
        'IPO': 'initial public offering',
        'M&A': 'mergers and acquisitions',
        'EBIT': 'earnings before interest and taxes',
        'IRR': 'internal rate of return',
        'ROA': 'return on assets',
        'ROE': 'return on equity',
        'NAV': 'net asset value',
        'PE ratio': 'price-to-earnings ratio',
        'EPS growth': 'earnings per share growth',
        'Fiscal Year': 'financial year',
        'CAPEX': 'capital expenditure',
        'APR': 'annual percentage rate',
        'P&L': 'profit and loss',
        'NPM': 'net profit margin',
        'EBT': 'earnings before taxes',
        'EBITDAR': 'earnings before interest, taxes, depreciation, amortization, and rent',
        'PAT': 'profit after tax',
        'COGS': 'cost of goods sold',
        'EBTIDA': 'earnings before taxes, interest, depreciation, and amortization',
        'E&Y': 'Ernst & Young',
        'B2B': 'business to business',
        'B2C': 'business to consumer',
        'LIFO': 'last in, first out',
        'FIFO': 'first in, first out',
        'FCF': 'free cash flow',
        'LTM': 'last twelve months',
        'OPEX': 'operating expenses',
        'TSR': 'total shareholder return',
        'PP&E': 'property, plant, and equipment',
        'PBT': 'profit before tax',
        'EBITDAR margin': 'earnings before interest, taxes, depreciation, amortization, and rent margin',
        'ROIC': 'return on invested capital',
        'EPS': 'earnings per share',
        'P/E': 'price-to-earnings',
        'EBITDA': 'earnings before interest, taxes, depreciation, and amortization',
        'YOY': 'year-over-year',
        'MOM': 'month-over-month',
        'CAGR': 'compound annual growth rate',
        'GDP': 'gross domestic product',
        'ROI': 'return on investment',
        'ROE': 'return on equity',
        'EBIT': 'earnings before interest and taxes',
        'DCF': 'discounted cash flow',
        'GAAP': 'Generally Accepted Accounting Principles',
        'LTM': 'last twelve months',
        'EBIT margin': 'earnings before interest and taxes margin',
        'EBT': 'earnings before taxes',
        'EBTA': 'earnings before taxes and amortization',
        'FTE': 'full-time equivalent',
        'EBIDTA': 'earnings before interest, depreciation, taxes, and amortization',
        'EBTIDA': 'earnings before taxes, interest, depreciation, and amortization',
        'EBITDAR': 'earnings before interest, taxes, depreciation, amortization, and rent',
        'COGS': 'cost of goods sold',
        'APR': 'annual percentage rate',
        'PESTEL': 'Political, Economic, Social, Technological, Environmental, and Legal',
        'KPI': 'key performance indicator',
        'SWOT': 'Strengths, Weaknesses, Opportunities, Threats',
        'CAPEX': 'capital expenditures',
        'EBITDARM': 'earnings before interest, taxes, depreciation, amortization, rent, and management fees',
        'EBITDAX': 'earnings before interest, taxes, depreciation, amortization, and exploration expenses',
        'EBITDAS': 'earnings before interest, taxes, depreciation, amortization, and restructuring costs',
        'EBITDAX-C': 'earnings before interest, taxes, depreciation, amortization, exploration expenses, and commodity derivatives',
        'EBITDAX-R': 'earnings before interest, taxes, depreciation, amortization, exploration expenses, and asset retirement obligations',
        'EBITDAX-E': 'earnings before interest, taxes, depreciation, amortization, exploration expenses, and environmental liabilities'
    
    # Add more abbreviations and replacements as needed
    }
    for abbreviation, full_form in replacements.items():
        text = text.replace(abbreviation, full_form)
    
    return text

def clean_and_preprocess(text):
    text=[x for x in text.split("\n") if len(x)>100]
    l=[]
    for t in text:
        # Convert to lowercase
        t = t.lower()    
        # Tokenize text into words
        words = nltk.word_tokenize(t)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]

        # Join the words back into a cleaned text
        cleaned_text = ' '.join(filtered_words)
        l.append(cleaned_text)
    return "\n".join(l)
def convert_amount_to_number(amount_str):
    try:
        return w2n.word_to_num(amount_str)
    except ValueError:
        return 0  # Return 0 if the conversion fails
def getTopic(encoded_input):
    # modelTopic.to("cuda")
    with torch.no_grad():
        logits = modelTopic(**encoded_input).logits
    predicted_class_id = logits.argmax().item()
    return modelTopic.config.id2label[predicted_class_id]
def selectedCorpusForNextQuarterModel(x,quarter,year):
    number_word_dict = {
    "1": "first",
    "2": "second",
    "3": "third",
    "4": "fourth",
    # Add more entries as needed
    }
    tokens=tokenizerTopic(x, padding=True, truncation=True, return_tensors='pt')
    splitSize=256
    chunksInput_ids=[tokens["input_ids"][0][r*splitSize:(r+1)*splitSize] for r in range(math.ceil(len(tokens["input_ids"][0])/splitSize))]
    chunksToken_type_ids=[tokens["token_type_ids"][0][r*splitSize:(r+1)*splitSize] for r in range(math.ceil(len(tokens["token_type_ids"][0])/splitSize))]
    chunksAttention_mask=[tokens["attention_mask"][0][r*splitSize:(r+1)*splitSize] for r in range(math.ceil(len(tokens["attention_mask"][0])/splitSize))]
    l=[]
    for idx in range(len(chunksInput_ids)):
        l.append({"input_ids":torch.tensor([list(chunksInput_ids[idx])]),
         "token_type_ids":torch.tensor([list(chunksToken_type_ids[idx])]),
          "attention_mask":torch.tensor([list(chunksAttention_mask[idx])])
        })
      
    selectedTopics = ["Stock Movement", "Earnings", "IPO", "Stock Commentary", "Currencies", "M&A | Investments", "Financials", "Macro", "Analyst Update", "Company | Product News"]
    result = [tokenizerTopic.decode(x["input_ids"][0], skip_special_tokens=True) for x in l if getTopic(x) in selectedTopics]
    result=[x for x in result if len(x)>10]
    des=f"the {number_word_dict[str(quarter)]} quarter results of the {year}"
    courpus=result
    embeddings_1 = similarityModel.encode([des]+courpus, normalize_embeddings=True,show_progress_bar=False) 
    sents=[des]+courpus
    rest=[sents[f] for f in [list(cosine_similarity(embeddings_1)[0][1:]).index(value)+1 for value in sorted(list(cosine_similarity(embeddings_1)[0][1:]),reverse=True)][:3]]
    return ",".join(rest)
def getQuarterPrediction(text):
    tokens=tokenizerQuarter(text,padding=True,max_length=512,return_overflowing_tokens=False,add_special_tokens=True,truncation=True,return_tensors="pt")
    with torch.no_grad():
        logits = modelQuarter(**tokens).logits
    predicted_class_id = logits.argmax().item()
    return modelQuarter.config.id2label[predicted_class_id]
def getSentence(listOfSentences,value):
    for sent in listOfSentences:
        if value in sent:
            return sent
    return value

def createExamples():
    # Create four buttons aligned horizontally
    col1, col2, col3, col4 = st.beta_columns(4)

    # Button 1
    if col1.button("Button 1"):
        transcript = read_and_display_text_file("file1.txt")

    # Button 2
    if col2.button("Button 2"):
        transcript = read_and_display_text_file("file2.txt")

    # Button 3
    if col3.button("Button 3"):
        transcript = read_and_display_text_file("file3.txt")

    # Button 4
    if col4.button("Button 4"):
        transcript = read_and_display_text_file("file4.txt")

    # Display the transcript
    if 'transcript' in locals():
       st.text_area("Enter the transcript:",transcript, height=100)


st.header("Transcript Analysis", divider='rainbow')
mainTranscript = st.text_area("Enter the transcript:", height=100)
createExamples()
doc = nlp(mainTranscript)
sentences = [sent.text for sent in doc.sents]

quarter= st.selectbox('Select your quarter',('1', '2', '3','4'))
year = st.selectbox('Select your quarter',tuple([str(x) for x in range(1900,int(datetime.datetime.now().year)+1,1)]))

if st.button("Analyze"):
    transcript=replace_abbreviations(mainTranscript)
    transcript=replace_abbreviations(transcript)
    transcript=removeSpeakers(transcript)
    transcript=removeQA(transcript)
    transcript=clean_and_preprocess(transcript)
    tokens=transcript.split()
    splitSize=256
    chunks=[tokens[r*splitSize:(r+1)*splitSize] for r in range(math.ceil(len(tokens)/splitSize))]
    chunks=[" ".join(chuk) for chuk in chunks]
    st.subheader("Management Sentiment", divider='rainbow')
    sentiment = [sentiment_model(x)[0]['label'] for x in chunks]
    sentiment=max(sentiment,key=sentiment.count)
    sentiment_color = "green" if sentiment == "postive" else "red"
    st.markdown(f'<span style="color:{sentiment_color}">{sentiment}</span>', unsafe_allow_html=True)
    st.subheader("Next Quarter Perdiction", divider='rainbow')
    # increase_decrease = [increase_decrease_model(x)[0]['label'] for x in chunks]
    increase_decrease=getQuarterPrediction(selectedCorpusForNextQuarterModel(mainTranscript,quarter,year))
    increase_decrease_color = "green" if increase_decrease == "Increase" else "red"
    st.markdown(f'<span style="color:{increase_decrease_color}">{increase_decrease}</span>', unsafe_allow_html=True)
    st.subheader("Financial Metrics", divider='rainbow')
    ner_result=[]
    savedchunks=[]
    idx=0
    while idx<len(chunks):
        ents=nlpPipe(chunks[idx])
        if len(ents)>=1:
            idxx=0
            savedchunks.append(idx)
            while idxx<len(ents):
                if len(ents[idxx]["word"].split())==2:
                    ner_result.append({ents[idxx]["entity_group"]:ents[idxx]["word"]})
                elif len(ents[idxx]["word"].split())==1:
                    try:
                        ner_result.append({ents[idxx]["entity_group"]:ents[idxx]["word"]+ents[idxx+1]["word"]+ents[idxx+2]["word"]})
                        idxx=idxx+2
                    except:
                        pass
                idxx=idxx+1
        idx=idx+1
    profits=[x["profit"] for x in ner_result if "profit" in x]
    revenues=[x["revenue"] for x in ner_result if "revenue" in x]
    expences=[x["expense"] for x in ner_result if "expense" in x]
    for idx in range(len(revenues)): 
         st.text_input(f'Revenue:{idx+1}', revenues[idx])
         st.text_input(f'Revenue-Sentence:{idx+1}', getSentence(sentences,revenues[idx]))
    for idx in range(len(profits)):
         st.text_input(f'Profit:{idx+1}', profits[idx])
         st.text_input(f'Profit-Sentence:{idx+1}', getSentence(sentences,profits[idx]))
    for idx in range(len(expences)):
         st.text_input(f'Expences:{idx+1}', expences[idx])
         st.text_input(f'Expences-Sentences:{idx+1}', getSentence(sentences,expences[idx]))

    st.subheader("Investment Recommendation", divider='rainbow')
    profitAmount=sum([convert_amount_to_number(x) for x in profits])
    expencesAmount=sum([convert_amount_to_number(x) for x in expences])
    if increase_decrease=="Increase" and sentiment=="postive" and profitAmount>expencesAmount:
        st.markdown(f'<span style="color:green">{"This is a great chance for investment. Do consider it."}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:red">{"Not the best chance for investment."}</span>', unsafe_allow_html=True)
