import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from string import punctuation
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import load_model
 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Student Courses Review App"

 
courses_df = pd.read_csv('./courses_data.csv')

 
app.layout = dbc.Container([
    html.H1("Courses Review App", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Courses")),
                dbc.CardBody([
                    dcc.Graph(id='courses-graph'),
                ]),
            ]),
        ], lg=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Student Reviews")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='course-review',
                        options=[{'label': course, 'value': course} for course in courses_df['Course']],
                        placeholder='Select an course to review'
                    ),
                    dbc.Input(id='student-name', type='text', placeholder='Your Name', className="my-2"),
                    dbc.Input(id='student-rating', type='number', placeholder='Rating (1-5)', className="my-2"),
                    dbc.Textarea(id='student-comment', placeholder='Review Comment', className="my-2"),
                    dbc.Button('Submit', id='submit-button', n_clicks=0, color="primary", className="my-2"),
                    html.Div(id='validation-message', className="text-danger"),   
                ]),
            ]),
        ], lg=6),
    ]),

    dbc.Card([
        dbc.CardHeader(html.H4("Insights")),
        dbc.CardBody([
            dcc.Graph(id='insights-graph'),
        ]),
    ], className="my-4"),

], fluid=True)





@app.callback(
    Output('courses-graph', 'figure'),
    Output('insights-graph', 'figure'),
    Output('course-review', 'value'),   
    Output('student-name', 'value'),   
    Output('student-rating', 'value'),   
    Output('student-comment', 'value'),   
    Output('validation-message', 'children'),   
    Input('submit-button', 'n_clicks'),
    State('course-review', 'value'),
    State('student-name', 'value'),
    State('student-rating', 'value'),
    State('student-comment', 'value')
)
def update_courses_and_insights(n_clicks, course_review, student_name, student_rating, student_comment):
    validation_message = ''   

    if n_clicks is None or n_clicks == 0:
        return dash.no_update, dash.no_update, course_review, student_name, student_rating, student_comment, validation_message
        
    if not course_review:
         
        validation_message = 'Please select a course.'
        return dash.no_update, dash.no_update, course_review, student_name, student_rating, student_comment, validation_message

    if not student_name:         
        validation_message = 'Please enter your name.'
        return dash.no_update, dash.no_update, course_review, student_name, student_rating, student_comment, validation_message

    if student_rating is None:         
        validation_message = 'Please enter a rating (1-5).'
        return dash.no_update, dash.no_update, course_review, student_name, student_rating, student_comment, validation_message

     
    if not (1 <= student_rating <= 5):
         
        validation_message = 'Rating must be between 1 and 5.'
        return dash.no_update, dash.no_update, course_review, student_name, student_rating, student_comment, validation_message

    course_index = courses_df.index[courses_df['Course'] == course_review].tolist()[0]
    sentiment = get_sentiment(student_comment)
    if sentiment == 'Positive':
        courses_df.at[course_index, 'Num_Positive'] += 1
    elif sentiment == 'Negative':
        courses_df.at[course_index, 'Num_Negative'] += 1
    else:
        courses_df.at[course_index, 'Num_Neutral'] += 1


    courses_df.at[course_index, 'Average Rating'] = (courses_df.at[course_index, 'Average Rating'] + student_rating) / 2
    courses_df.at[course_index, 'Sentiment'] = sentiment  


    courses_df.to_csv('courses_data.csv', index=False)   


    courses_fig = px.bar(courses_df, x='Course', y='Average Rating', title='Course Ratings')
    sentiment_fig = create_sentiment_bar_chart(courses_df, course_review)
     
    course_review = ''
    student_name = ''
    student_rating = ''
    student_comment = ''

    return courses_fig, sentiment_fig, course_review, student_name, student_rating, student_comment, validation_message

def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


def create_sentiment_bar_chart(df, course):
    course_df = df[df['Course'] == course]
    # print(course)
    
    
    sentiment_counts = {
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [
            course_df['Num_Positive'].values[0],
            course_df['Num_Negative'].values[0],
            course_df['Num_Neutral'].values[0]
        ]
    }

    # print(sentiment_counts)
    
    
    sentiment_fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        labels={'x': 'Sentiment', 'y': 'Count'},
        title=f'Sentiment Analysis for {course}'
    )
    
    return sentiment_fig




# Load the model
model = load_model('/content/sentiment/LSTM.h5')

# Define preprocessing functions
def preprocess_text(text):
    # Making the words lowercase
    text = text.lower()
    # Removing characters
    text = ''.join([c for c in text if c not in punctuation])
    # Removing stopwords 
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


def get_sentiment(text):
    # Apply preprocessing to each text element in the list
    preprocessed_text = [preprocess_text(text)]

    # Create a tokenizer and fit it on the preprocessed text
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(preprocessed_text)

    # Tokenize and vectorize the text
    sequences = tokenizer.texts_to_sequences(preprocessed_text)

    # Pad the sequences to match the model's input shape
    max_sequence_length = 127  # Adjust this based on the model's input shape
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    # Make predictions (you need to load your model here)
    predictions = model.predict(padded_sequences)

    return predictions



if __name__ == '__main__':
    app.run_server(debug=True)