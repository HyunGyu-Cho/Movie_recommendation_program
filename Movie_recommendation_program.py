import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

class MovieRecommender:
    def __init__(self, api_key, language='ko-KR'):
        self.movie = Movie()
        self.tmdb = TMDb()
        self.tmdb.api_key = api_key
        self.tmdb.language = language

    def get_recommendations(self, title, movies, cosine_sim):
        idx = movies[movies['title'] == title].index[0]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]

        images = []
        titles = []
        for i in sim_scores:
            id = movies['id'].iloc[i[0]]
            details = self.movie.details(id)

            poster_path = details.get('poster_path')
            image_path = self.get_image_path(poster_path)

            images.append(image_path)
            titles.append(details['title'])

        return images, titles

    def get_image_path(self, poster_path):
        base_url = 'https://image.tmdb.org/t/p/w500'
        return base_url + poster_path if poster_path else 'C:\\Cho\\Assignment\\OpenSourceSoftWare\\Project\\no_image.jpg'

def main():
    # MovieRecommender 인스턴스 생성
    recommender = MovieRecommender(api_key='abd433a0c39e16b0543d772acebd938e')

    # 데이터 로드
    movies = pickle.load(open('C:\\Cho\\Cho coding-project\\Project\\movies.pickle', 'rb'))
    cosine_sim = pickle.load(open('C:\\Cho\\Cho coding-project\\Project\\cosine_sim.pickle', 'rb'))

    # Streamlit UI 설정
    st.set_page_config(layout='wide')
    st.header('Movie Recommendation Program')

    # 영화 선택 드롭다운
    title = st.selectbox('Choose a movie you like', movies['title'].values)

    # 추천 버튼
    if st.button('Recommend'):
        # 추천 목록 가져오기
        with st.spinner('Please wait...'):
            images, titles = recommender.get_recommendations(title, movies, cosine_sim)

            # 2행 5열의 형태로 이미지 및 제목 출력
            idx = 0
            for _ in range(2):
                cols = st.columns(5)
                for col in cols:
                    col.image(images[idx])
                    col.write(titles[idx])
                    idx += 1

if __name__ == '__main__':
    main()
