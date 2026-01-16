# Link Prediction Challenge: MovieLens 100k

## Dependencies
Deepends on python3.10

`pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

`pip install dgl==2.1.0`

`pip install torchdata`

`pip install pandas numpy scikit-learn matplotlib seaborn tqdm`

## Dataset
Consists fo 3 tables:
* ratings
-------------------------------------
| user_id | movie_id | rating | timestamp|
--------------------------------------

* users
-------------------------------------
|user_id | age gender  |occupation |zip_code|
--------------------------------------

* movies
-------------------------------------
| movie_id | title | release_date | video_release_date | IMDb_URL|
-------------------------------------
