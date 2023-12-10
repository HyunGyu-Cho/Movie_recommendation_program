# 영화 추천 프로젝트

영화 추천 프로젝트에 오신 여러분을 환영합니다! 본 프로젝트는 TMDB API를 활용하여 영화 정보를 가져와 이를 기반으로 영화를 10개 추천해주는 시스템을 개발한 프로젝트입니다.

<br>

## 프로젝트 시작에 앞서
### 1. 참고 자료
이 프로그램은 다음의 두 가지 자료를 참고하여 제작되었습니다:

1. **TMDB 5000의 IBTESAM AHMED의 "Getting Started with a Movie Recommendation System"**
   - [TMDB5000-IBTESAM AHMED의 자료](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/notebook)

2. **나도코딩의 "파이썬 코딩 무료 강의- 머신러닝, 영화 추천 시스템 만들기"**
   - [나도코딩 유튜브 강의](https://www.youtube.com/watch?v=TNcfJHajqJY&t=22811s)

위의 두 가지 자료를 참고하여 프로그램이 제작되었습니다.

### 2. TMDB API key 생성
프로젝트를 실행하려면 TMDB에서 API 키를 발급받아야 합니다. 아래 단계를 따라 진행하세요:

1. TMDB 계정 생성: [TMDB 가입](https://www.themoviedb.org/signup)
2. API 키 생성: [TMDB API 키](https://www.themoviedb.org/settings/api)

### 3. 라이선스

이 프로젝트는 MIT 라이선스(LICENSE) 하에 배포됩니다.

<br>

## 프로젝트 개요
이 프로젝트는 영화의 줄거리(description)를 기반으로 모든 영화에 대한 (similarity) 유사도 점수를 계산하고, 그 유사도 점수를 기반으로 영화를 추천할 것입니다.<br>

이 프로그램에서는 텍스트 데이터를 벡터로 표현하기 위해 TF-IDF 기법을 사용합니다.

### <TF-IDF (Term Frequency-Inverse Document Frequency)에 대한 간단한 설명>

TF-IDF는 텍스트 데이터에서 각 단어의 중요도를 계산하는 기법으로, 정보 검색 및 텍스트 마이닝 분야에서 널리 사용됩니다.<br> 

이 기법은 문서 내에서 특정 단어의 상대적인 빈도를 나타내는 TF(Term Frequency)와 전체 문서 집합에서의 단어의 중요도를 나타내는 IDF(Inverse Document Frequency)를 결합하여 계산됩니다.

#### TF (Term Frequency)

TF는 특정 단어의 상대적인 빈도를 나타내는 지표입니다. <br>
각 단어의 TF는 해당 단어의 문서 내 출현 횟수를 전체 단어의 수로 나눈 값으로 표현됩니다.

#### IDF (Inverse Document Frequency)

IDF는 전체 문서 집합에서 특정 단어의 중요도를 나타내는 지표입니다. <br>
해당 단어를 포함한 문서의 수를 전체 문서 수로 나눈 뒤 로그를 취한 값으로 표현됩니다.

#### TF-IDF 계산

각 단어의 TF와 IDF를 곱하여 TF-IDF 값을 계산합니다. 이 값은 각 단어의 문서 내에서의 중요성을 고려한 상대적인 가중치를 나타냅니다.

#### 사용법

TF-IDF를 사용하여 문서 간 유사성을 측정하거나, 머신러닝 모델에 입력 데이터로 활용할 수 있습니다. <br>
scikit-learn의 TfIdfVectorizer 클래스를 사용하면 간편하게 TF-IDF 행렬을 생성할 수 있습니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 벡터라이저 객체 정의
tfidf = TfidfVectorizer()

# 데이터를 피팅하고 변환하여 TF-IDF 행렬 생성
tfidf_matrix = tfidf.fit_transform(data)
```
위의 설명은 간략한 예시일 뿐이며, 더 자세한 내용은 [링크](https://wikidocs.net/31698)를 참조하시기 바랍니다.

우리의 프로젝트에서는 4800개 영화를 설명하기 위해 20,000여 개의 다른 단어가 사용되어 줄거리를 설명하고 있습니다.<br>

따라서 4800x(20000+a)의 행렬이 생성될 수 있으며, 이 행렬을 바탕으로 유사도를 계산할 것입니다.<br>

이제, 우리는 두 영화 간의 유사성을 나타내는 숫자를 계산하기 위해 코사인 유사도를 사용할 것입니다.<br>

코사인 유사도를 구하는 식은 [링크](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F0pBN5%2FbtqBW1nWMbL%2FE4Hh4JpyOqT3Opg4MhPa70%2Fimg.png)를 참조해주시기 바랍니다.<br>

TF-IDF 벡터화를 사용하여 얻은 벡터들 간의 코사인 유사도를 계산할 때, 점곱(dot product)을 통해 바로 유사도 점수를 얻을 수 있습니다. <br>또한, 계산 속도가 더 빠른 linear_kernel() 함수를 사용하겠습니다.<br> 

linear_kernel() 함수는 두 벡터 간의 내적(점곱)을 계산하여 유사도를 측정하는 함수로, 특히 코사인 유사도를 계산할 때 효율적으로 사용됩니다.

```python
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

다음과 같은 작업이 끝나면 이제는 영화 제목을 입력값으로 받아 10개의 가장 유사한 영화 목록을 출력하는 함수를 정의할 것입니다.
``` python
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
```

예를 들어 input 값이 다음과 같다고 하면
```python
get_recommendations('Avatar', movies, cosine_sim)
```

output 값은 다음과 같습니다.
```python
65                              The Dark Knight
299                              Batman Forever
428                              Batman Returns
1359                                     Batman
3854    Batman: The Dark Knight Returns, Part 2
119                               Batman Begins
2507                                  Slow Burn
9            Batman v Superman: Dawn of Justice
1181                                        JFK
210                              Batman & Robin
```
즉, 위 코드의 결과값으로 나온 10개의 [index, title]정보가 바로 cos 유사도를 바탕으로 추천된 10개의 영화 정보입니다.<br> 
이 10개의 영화에 대해 image와 title을 받아서 2행 5열로 정렬시켜 그 값을 return해주는 것이 함수의 역할입니다. 

위 과정을 요약하면 다음과 같습니다<br>

1. 영화 제목을 주면 해당 영화의 인덱스를 얻습니다. <br>
2. 해당 영화와 다른 모든 영화 간의 코사인 유사도 점수 목록을 얻습니다. 이를 해당 영화의 위치와 유사도 점수가 튜플 형태로 담긴 리스트로 변환합니다.<br>
3. 유사도 점수를 기준으로 상기된 튜플 리스트를 정렬합니다. 즉, 두 번째 요소를 기준으로 정렬합니다.<br>
4. 이 리스트에서 상위 10개의 요소를 얻습니다. 첫 번째 요소는 자기 자신에 해당하므로 무시합니다.<br>
5. 최상위 요소들의 인덱스에 해당하는 영화 제목들을 반환합니다."<br>

#### TMDB의 database를 쉽게 사용할 수 있는 라이브러리를 사용
위 라이브러리를 이용해서 하나의 클래스를 생성해 TMDB에서 제공하는 기능을 더욱 더 쉽고 간단하게 사용할 수 있습니다.<br>
다음과 같은 간단한 코드 작성으로 말이죠.
```python
from tmdbv3api import Movie, TMDb
def __init__(self, api_key, language='ko-KR'):
        self.movie = Movie()
        self.tmdb = TMDb()
        self.tmdb.api_key = api_key
        self.tmdb.language = language
```

#### pickle 라이브러리의 사용
```python
import pickle
```

우리는 또한 pickle 라이브러리를 사용해서 TMDB에서 사용하는 데이터베이스에 기반한 pickle 파일을 만들어 별도의 과정을 거치지 않고 이미 직렬화된 객체를 불러올 수 있습니다.   
``` python
movies = pickle.load(open('파일경로\\movies.pickle', 'rb'))
cosine_sim = pickle.load(open('파일경로\\cosine_sim.pickle', 'rb'))
```

#### streamlit을 이용한 간단한 웹 페이지의 제작
이제 우리는 Streamlit을 사용하여 간단한 영화 추천 웹 애플리케이션을 만드는 과정을 살펴볼 것입니다. <br>

우선, Streamlit은 데이터 과학 및 머신러닝 프로젝트를 위한 웹 애플리케이션을 간편하게 만들 수 있는 Python 라이브러리입니다.<br>
```python
import streamlit as st
```
다음과 같은 코드를 입력해 streamlit 라이브러리를 가져옵니다.<br>
<br>
이제 명령 프롬프트 창에 python 코드가 있는 폴더로 경로이동을 한 뒤, "streamlit run python파일명"을 입력해 python 파일을 실행시킵니다.<br>

python파일이 실행되면 웹 페이지가 열립니다.<br>

웹페이지가 열리면 "Movie Recommendation Program"이라는 헤더가 표시됩니다.<br>

드롭다운 메뉴를 사용하여 사용 가능한 옵션에서 영화를 선택합니다.<br>

"Recommend" 버튼을 클릭하여 선택한 영화를 기반으로 추천 영화 목록을 가져옵니다.<br>

웹페이지는 추천된 영화를 나타내는 2행 5열의 그리드로 구성된 포스터 및 제목을 표시합니다.<br>

사용자가 입력한/선택한 영화의 콘텐츠 기반 필터링을 바탕으로 cos유사도가 높은 10개의 영화가 추천되어 보여질 것입니다. <br>
<br>
<br>


## 마무리하며
다시한번 강조해서 말하지만, 위 코드는 사실상 나도코딩의 "파이썬 코딩 무료 강의- 머신러닝, 영화 추천 시스템 만들기" 를 모방하다시피 만들었습니다. <br>

첫 프로젝트인 만큼, 코드의 구현에 집착하기보다는 머신러닝이 어떻게 실제 프로젝트에 구현되는지에 집중했습니다.<br>

<TF-IDF>직렬화에 대해서 자세히 알아보고 배웠으며, cos유사도의 정의, 계산법에 대해서 살펴보았습니다. <br>

마지막으로 streamlit을 이용한 웹 페이지의 제작이 어떻게 이루어지는지도 배워볼 수 있는 시간이었습니다. <br>

HTML, CSS, JS를 이용한 웹 페이지 제작 이외의 방법이 있다는 것에 신기했습니다.<br>

이 프로젝트를 시작으로, 앞으로는 TMDB 5000의 IBTESAM AHMED의 "Getting Started with a Movie Recommendation System" 에서 다루고 있는 인구 통계학적 필터링, 협업 필터링을 기반으로 한 영화 추천 웹페이지를 만들어 볼 것입니다.<br>

머신러닝을 공부하면서 배운 다양한 수학적 개념들이 정말 중요하다는 사실을 확실하게 깨달았습니다!




