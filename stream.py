from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import torch
import json
import logging
import os

import streamlit as st
import wikipedia
from typing import Dict
import random

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

model = QuestionAnsweringModel(
    'distilbert',
    'official-default2-epoch-3',
    use_cuda=False
)

@st.cache(suppress_st_warning=True)
def answer_question(context, question):
    to_predict = [{
        'context': context,
        'qas': [{
        'id': str(random.randint(2, 5000)),
        'question': question,
        'context': context
        }]
    }]
    answers, probabilities = model.predict(to_predict, n_best_size=10)
    answers = [answer['answer'] for answer in answers]
    if len(answers[0]) > 1:
        return answers[0][0] if answers[0][0] not in ['empty', ''] else answers[0][1]
    else:
        return answers[0][0]

@st.cache(suppress_st_warning=True)
def get_wiki_paragraph(query):
    search = wikipedia.search(query)
    try:
        summary = wikipedia.summary(search[0], chars=384)
    except wikipedia.DisambiguationError as err:
        ambiguous_terms = err.options
        return wikipedia.summary(ambiguous_terms[0], chars=384)
    return summary

if __name__ == "__main__":

    title_slot = st.title('Information retrieval')
    paragraph_slot = st.empty()
    help_text = 'Provide a context for your question'
    original_context = st.text_area('Original Text', height=3, help=help_text)
    question = st.text_input("QUESTION", help='Ask a question', max_chars=128)

    if st.checkbox('Use Wikipedia'):
        wiki_query = st.text_input("WIKIPEDIA SEARCH TERM", "")

        if wiki_query:
            wiki_para = get_wiki_paragraph(wiki_query)
            paragraph_slot.markdown(wiki_para)
            title_slot.markdown('# Wikipedia Article')

            # Execute question against paragraph
            if question != '':
                try:
                    with st.spinner(text='Searching for answer...🧐👀⌚⏳⌛'):
                        answer = answer_question(wiki_para, question)
                        if answer:
                                st.success(answer)
                        else:
                                st.info('Sorry, I do not have any answer to this question')
                except Exception as e:
                    print(e)
                    st.warning("You must provide a valid wikipedia paragraph")
            else:
                st.warning('Kindly provide a question')

    elif original_context:
        paragraph_slot.markdown(original_context)
        title_slot.markdown('# Original Article')
        # Execute question against paragraph
        if question != "":
            try:
                with st.spinner(text='Searching for answer...🧐👀⌚⏳⌛'):
                    answer = answer_question(original_context, question)
                    if answer:
                        st.success(answer)
                    else:
                        st.info('Sorry, I do not have any answer to this question')
            except Exception as e:
                print(e)
                st.warning("You must provide a valid wikipedia paragraph")
