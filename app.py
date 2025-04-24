from flask import Flask, render_template, Response
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
DATA_PATH = r"Fake_Extended.csv"  # Use raw string to avoid escape sequence issues
df = pd.read_csv(DATA_PATH)

# Preprocess columns
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['text'].apply(lambda x: len(str(x)))
df['title_length'] = df['title'].apply(lambda x: len(str(x)))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

def render_plot(fig):
    img = BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return Response(img.getvalue(), mimetype='image/png')


@app.route('/wordcloud')
def wordcloud():
    text = ' '.join(df['title'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of News Titles")
    return render_plot(fig)

@app.route('/timeseries')
def timeseries():
    df_time = df.dropna(subset=['date']).groupby(df['date'].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(12, 5))
    df_time.plot(ax=ax, marker='o', color='blue')
    ax.set_xlabel("Date (Monthly)")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Articles Published Over Time")
    ax.grid()
    return render_plot(fig)

@app.route('/monthly_articles')
def monthly_articles():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='month', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Articles Per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    return render_plot(fig)

@app.route('/heatmap')
def heatmap():
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[['word_count', 'char_count', 'title_length']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    return render_plot(fig)

@app.route('/piechart')
def piechart():
    fig, ax = plt.subplots(figsize=(8, 8))
    df['subject'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=ax)
    ax.set_title("Articles by Subject")
    ax.set_ylabel("")
    return render_plot(fig)

@app.route('/barplot')
def barplot():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='subject', y='word_count', hue='subject', palette='magma', dodge=False, ax=ax)
    ax.set_title("Subject vs Word Count")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Word Count")
    plt.xticks(rotation=45)
    return render_plot(fig)

@app.route('/scatterplot')
def scatterplot():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='word_count', y='char_count', hue='subject', palette='coolwarm', ax=ax)
    ax.set_title("Word Count vs Char Count")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Character Count")
    return render_plot(fig)

@app.route('/pairplot')
def pairplot():
    fig = sns.pairplot(df[['word_count', 'char_count', 'title_length']], diag_kind='kde', plot_kws={'s': 10}).fig
    fig.suptitle("Pair Plot of Numeric Features", y=1.02)
    return render_plot(fig)

if __name__ == '__main__':
    app.run(debug=True)