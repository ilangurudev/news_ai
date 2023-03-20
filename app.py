import streamlit as st
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAIChat
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain


# Function to generate empty dataframe
def create_empty_dataframe(rows, columns):
    data = {column: [""] * rows for column in columns}
    return pd.DataFrame(data)


# Sidebar
st.sidebar.title("Configuration")
selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "chat-gpt-3.5-turbo",
    ],  # "gpt3-text-davinci-003"
)
selected_dataset = st.sidebar.selectbox(
    "Select Dataset", ["Regulatory Changes", "Dealer Risk", "Custom"]
)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


prompt_prefix = """You are a news recommender and your job is to look at an article and decide if I would find it useful or not.
{dataset_specific_prompt}
I'm going to pass in a few headlines and publishers. Can you just respond with "Relevant" or "Not Relevant" for the headlines I send you?"""

if selected_dataset == "Regulatory Changes":
    dataset_specific_prompt = """I work in a bank/credit card company and I'm looking for news articles that deal with changes in the regulatory environment. 
It could be articles about regulatory authorities like CFPB or OCC announcing changes in regulation or a new focus areas. 
Or articles like the president or senate or government announcing new regulations. 
Or peer financial institutions getting fined for violations. 
To reiterate I'm especially looking for articles in the finance, fin-tech, banking and credit card sector. 
**I don't need think pieces or opinion pieces or editorials that talk about the consequences of a regulation but just the ones that talk about objective changes in the regulatory environment that might impact my work**"""
    examples = [
        {
            "headline": "CFPB Launches Inquiry Into the Business Practices of Data Brokers",
            "publisher": "CFPB",
            "answer": "Relevant",
        },
        {
            "headline": "OCC Issues Prohibition Order, Fines Former Wells Fargo Executive $17 Million in Settlement",
            "publisher": "NYT",
            "answer": "Relevant",
        },
        {
            "headline": "CFPB Announces Appointments of New Advisory Committee Members",
            "publisher": "CFPB",
            "answer": "Not Relevant",
        },
    ]
    prediction_examples = [
        {
            "headline": "Ray Dalio Commentary: What I Think About the Silicon Valley Bank Situation",
            "publisher": "Yahoo Entertainment",
        },
        {
            "headline": "Financial Regulators, Black History and Epistemic Capital",
            "publisher": "Harvard School of Engineering",
        },
        {
            "headline": "Agencies Issue Joint Statement on Crypto-Asset Risks to Banking Organizations",
            "publisher": "OCC",
        },
        {
            "headline": "Innovating during a regulatory wave",
            "publisher": "Venture beat",
        },
        {
            "headline": "CFPB Proposes Rule to Rein in Excessive Credit Card Late Fees",
            "publisher": "CFPB",
        },
    ]

elif selected_dataset == "Dealer Risk":
    dataset_specific_prompt = """I work in a auto-loan company and I'm looking for news articles that talk about with automobile dealerships being in the news for committing fraud and crime.
I am not interested in just about any crime or fraud but only that which is related to dealerships. 
I am NOT concerned about articles that where the dealership is the victim of a crime. 
I am only interested where the dealerships are the perpetrators of said crime or accusations.
"""
    examples = [
        {
            "headline": "Kentucky Police arrest suspect following $600,000 multiple car heist that took less than 45 seconds",
            "publisher": "Business Insider",
            "answer": "Not Relevant",
        },
        {
            "headline": "Car Dealership workers kidnapped, tortured in Armed Robbery",
            "publisher": "ABC7 Chicago",
            "answer": "Not Relevant",
        },
        {
            "headline": "Auto Dealership Owner Sentenced to 10 Years in Prison for $1.5 Million Fraud Scheme",
            "publisher": "Long Island Business News",
            "answer": "Relevant",
        },
    ]
    prediction_examples = [
        {
            "headline": "New York City Pharmacist charged with running healthcare fraud scheme",
            "publisher": "amnycom",
        },
        {
            "headline": "California man arrested for 3.6 million paycheck protection program fraud",
            "publisher": "DoJ",
        },
        {
            "headline": "Former Moline Businessman sentenced to 41 months in connection with scam to rollback odometeres at used car dealerships",
            "publisher": "DoJ",
        },
        {
            "headline": "Kentucky car dealer convicted in truck warranty fraud scheme",
            "publisher": "apnews.com",
        },
        {
            "headline": "Owner of car dealership pleads guilty to defraduing financing company",
            "publisher": "Automotive News",
        },
    ]

elif selected_dataset == "Custom":
    dataset_specific_prompt = """I work in a XYZ company and I'm looking for news articles that deal with XYZ"""
    examples = [{k: "" for k in ["headline", "publisher", "answer"]} for i in range(3)]
    prediction_examples = [{k: "" for k in ["headline", "publisher"]} for i in range(5)]


prompt_prefix = prompt_prefix.format(
    dataset_specific_prompt=dataset_specific_prompt
).replace("\n", " ")


# Main panel
st.title("News Topic Subscription")

st.header("Dataset")
st.write(f"Prompt data for {selected_dataset}:")

prompt_data = st.text_area(
    "Prompt Prefix", prompt_prefix, placeholder="Enter dataset prompt here", height=300
)

st.header("Few Shot Examples")

columns = ["headline", "publisher", "answer"]
data = pd.DataFrame({k: v for k, v in example.items()} for example in examples)
data.columns = columns
# data["Relevance"] = [""] * 3

for index, row in data.iterrows():
    row["headline"] = st.text_input(
        f"Headline", row["headline"], key=f"headline_{index}"
    )
    row["publisher"] = st.text_input(
        f"Publisher", row["publisher"], key=f"publisher_{index}"
    )
    choice_index = (
        ["Relevant", "Not Relevant"].index(row["answer"]) if row["answer"] != "" else 0
    )
    row["answer"] = st.radio(
        f"Relevance {index + 1}",
        index=choice_index,
        options=["Relevant", "Not Relevant"],
        key=f"answer_{index}",
    )
    st.markdown("---")

st.header("Predictions")

predictions_data = pd.DataFrame(
    {k: v for k, v in example.items()} for example in prediction_examples
)
predictions_data["answer"] = [""] * 5
prediction_labels = []

for index, row in predictions_data.iterrows():
    row["headline"] = st.text_input(
        f"headline", row["headline"], key=f"prediction_headline_{index}"
    )
    row["publisher"] = st.text_input(
        f"publisher", row["publisher"], key=f"prediction_publisher_{index}"
    )
    prediction_labels.append(st.empty())
    st.markdown("---")


llm = OpenAIChat(temperature=0.0, openai_api_key=openai_api_key)
example_prompt = PromptTemplate(
    input_variables=["headline", "publisher", "answer"],
    template="Headline: {headline}\nPublisher: {publisher}\nAnswer: {answer}",
)


def update_relevance(predictions_data):

    prompt = FewShotPromptTemplate(
        prefix=prompt_data,
        examples=data.to_dict("records"),
        example_prompt=example_prompt,
        suffix="Headline: {headline}\nPublisher: {publisher}\nAnswer: ",
        input_variables=["headline", "publisher"],
    )
    print(prompt.format(headline="", publisher=""))
    print(data)
    news_predictor_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    token_counts = []

    for index, row in predictions_data.iterrows():
        if row["headline"] != "":
            with get_openai_callback() as cb:
                predictions_data.loc[index, "answer"] = news_predictor_chain.run(
                    {
                        "headline": row["headline"],
                        "publisher": row["publisher"]
                        if row["publisher"] != ""
                        else "Unknown",
                    }
                )
                token_counts.append(cb.total_tokens)

        else:
            predictions_data.loc[index, "answer"] = ""
    st.text(
        f"{sum(token_counts)} tokens used in this run for {len(token_counts)} articles. This costed ${sum(token_counts) * 0.002/1000 } @$0.002/1K tokens."
    )
    st.text(
        f"For predicting 1000 articles, it would cost ${sum(token_counts)/len(token_counts) * 0.002 } at the current rate. "
    )
    return predictions_data


if st.button("Run Model", type="primary"):
    predictions_data = update_relevance(predictions_data)
    for index, row in predictions_data.iterrows():
        if row["answer"] != "":
            with prediction_labels[index].container():
                if row["answer"] == "Relevant":
                    st.markdown("This article is **Relevant** :thumbsup:")
                else:
                    st.markdown("This article is **Not Relevant** :thumbsdown:")
