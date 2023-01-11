import pickle
import numpy as np
import time
import streamlit as st

book_df = pickle.load(open("books.pkl", "rb"))
model = pickle.load(open("structure.pkl", "rb"))

books = []
for i in range(book_df.shape[0]):
    books.append(book_df.index[i])


def recommend_book(book_name):
    recommended_books = []

    book_id = np.where(book_df.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_df.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    for i in range(len(suggestions)):
        recommended_books.append(book_df.index[suggestions[i]])
    for i in range(len(recommended_books[0])):
        if i != 0:
            st.subheader(str(i) + ") " + recommended_books[0][i])


# ui model
st.title("Book Recommendation System")
input_book = st.selectbox("Books", books)
clicked = st.button("Recommend")

if clicked:
    with st.spinner('Wait for it...'):
        time.sleep(0.8)
    st.success('Done!')
    recommend_book(input_book)
    st.snow()
