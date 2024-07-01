import streamlit as st


def levenshtein_distance(token1, token2):
    distances = [[0] * (len(token2) + 1) for _ in range(len(token1) + 1)]
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                distances[t1][t2] = min(
                    distances[t1][t2 - 1] + 1, distances[t1 - 1][t2] + 1, distances[t1 - 1][t2 - 1] + 1)
    return distances[len(token1)][len(token2)]


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words


vocabs = load_vocab(file_path='./vocab.txt')
vocabs = [line.split() for line in vocabs]

st.title("Word Correction using Levenshtein Distance")

if 'times' not in st.session_state:
    st.session_state.times = 0

if 'check' not in st.session_state:
    st.session_state.check = False

if 'correct_word' not in st.session_state:
    st.session_state.correct_word = ''

if 'vocab_meanings' not in st.session_state:
    st.session_state.vocab_meanings = {}

if 'sorted_distances' not in st.session_state:
    st.session_state.sorted_distances = {}

word = st.text_input('Word:', value=st.session_state.get('word_input', ''))

if st.button("Compute") and word:
    st.session_state.word_input = word
    leven_distances = {vocab[0]: levenshtein_distance(
        word, vocab[0]) for vocab in vocabs}
    st.session_state.vocab_meanings = {
        vocab[0]: " ".join(vocab[2:]) for vocab in vocabs}
    st.session_state.sorted_distances = dict(
        sorted(leven_distances.items(), key=lambda item: item[1]))
    st.session_state.correct_word = list(
        st.session_state.sorted_distances.keys())[0]

    st.session_state.times = 0  # Reset the retry counter on new computation

if 'word_input' in st.session_state:
    st.write('Correct word: ', st.session_state.correct_word)
    st.write(
        'Meaning: ', st.session_state.vocab_meanings[st.session_state.correct_word])

    col1, col2 = st.columns(2)
    col1.write('Vocabulary:')
    col1.write(st.session_state.vocab_meanings)
    col2.write('Distances:')
    col2.write(st.session_state.sorted_distances)

    if st.session_state.word_input != st.session_state.correct_word:
        st.write(
            'Don\'t look at the previous answer and type again a valid word. Let\'s see if you can write it correctly this time.')
        new_word = st.text_input('Another trial:', key='new_word_input')

        if st.button('New'):
            if st.session_state.times < 5:
                if new_word in st.session_state.vocab_meanings.keys():
                    st.write('You did a great job.')
                    st.session_state.check = True
                else:
                    st.session_state.times += 1
                    st.write(
                        f'Your word is invalid. Your {st.session_state.times}st trial is done. Please try again.')
            if st.session_state.times >= 5:
                st.write('You need to practice more. Try again next time.')
            if st.session_state.times >= 5 and st.session_state.check:
                st.write(
                    'Your performance has been improved. Congratulations!!!')
