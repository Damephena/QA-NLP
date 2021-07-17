from simpletransformers.question_answering import QuestionAnsweringModel
import torch
import logging

import streamlit as st
import wikipedia
import random

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

has_cuda = torch.cuda.is_available()
model = QuestionAnsweringModel(
    'distilbert',
    'Ifenna/dbert-3epoch',
    use_cuda=has_cuda
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

def answer_search(context, question):

    '''Animated spinner during model prediction'''
    try:
        with st.spinner(text='Searching for answer...🧐👀⌚⏳⌛'):
            answer = answer_question(context, question)
            if answer != 'empty':
                st.success(answer)
            else:
                st.info('Sorry, I do not have any answer to this question')
    except Exception as e:
        st.warning('You must provide a valid original paragraph')

def check_question(context, question):

    '''Checks that the question textbox is not empty.'''
    if question != '':
        answer_search(context, question)
    else:
        st.error('Ask me a question.')

def main():
    subheader_1 = '''
    This is a simple user interface that implements the custom Question Answering model by Ifenna.
    - Provide a context for your question; *original text* or *using wikipedia.*
    - Ask a question based on the context you provided.
    - Wait for a response from the system which is predicted by the custom model.
    ---
    '''
    title_slot = st.title('Information retrieval')
    subheader_slot = st.markdown(subheader_1)
    paragraph_slot = st.empty()
    question = st.text_input('QUESTION', help='Ask a question', max_chars=128)

    if st.checkbox('Use Wikipedia'):
        wiki_query = st.text_input('WIKIPEDIA SEARCH TERM', '')

        if wiki_query:
            wiki_para = get_wiki_paragraph(f'---\n{wiki_query}\n ---')
            paragraph_slot.markdown(wiki_para)
            title_slot.markdown('# Wikipedia Article')
            subheader_slot.markdown('*Article sourced from Wikipedia*')
            check_question(wiki_para, question)
        else:
            st.error('Kindly provide a valid Wikipedia term to search.')

    else:
        help_text = 'Provide a context for your question'
        original_context = st.text_area('Original Text', height=4, help=help_text)
        if original_context != '':
            paragraph_slot.markdown(f'---\n{original_context}\n ---')
            title_slot.markdown('# Original Article')
            subheader_slot.markdown('*Your Original article*')
            check_question(original_context, question)
        else:
            st.error('Kindly provide a context (original text) for your question')

if __name__ == '__main__':
    main()
