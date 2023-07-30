from flask import Flask, render_template, request
import requests
import csv
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import nltk
import datetime
import time
nltk.download("stopwords")
from nltk.corpus import stopwords
from collections import defaultdict
import math
import pandas as pd
from IPython.display import display, HTML

app = Flask(__name__)

def crawler():
    coventry_url = "https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/"
    all_data = []
    data = []
    for page_num in range(0, 10):  
        url = coventry_url + f"?page={page_num}"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
    
        for i in range(55):  
            classes = f"list-result-item list-result-item-{i}"
            results = soup.find_all('li', class_=classes)
            for result in results:
            #Name and links of authors
                author_names = []
                author_links = []
                all_authors = result.find_all("a", class_="link person", attrs={'span': ''})
                for author in all_authors:
                    author_names.append(author.text.strip())
                    author_links.append(author['href'])
                
            #Authors with no link
                non_cov_auth_names = []
                span_nolink_names = result.find_all('span')
                for name in span_nolink_names:
                    if name.string and ', ' in name.string:
                        author_name = name.string.strip()
                        if len(author_name) <= 20:
                             if not any(author_name in ad for ad in author_names):
                                non_cov_auth_names.append(author_name)
            
            #Title and link of publications
                titles =result.find('h3', class_='title')
                title_link=titles.a
                title =title_link.span.text.strip()
                title_url =title_link['href']
            
            #Date of publication
                dates= result.find('span', class_='date')
                date = dates.text.strip()
            
            #Combine all auhtors and remove duplicates
                all_authors = list(set(author_names + non_cov_auth_names))
                all_data.append([", ".join(all_authors), ",  ".join(author_links), title, title_url, date])

    filename = "important_data6.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(["Authors", "Authors_link", "Title", "Title_link", "Date"])
        writer.writerows(all_data)
    
    #print(f"Imported data were saved to {filename} successfully.")
    
    with open('important_data6.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Filter the data to remove rows with null values
    filtered_data = [row for row in data if all(value for value in row.values())]

    # Write the filtered data back to a new CSV file or overwrite the existing one
    with open('important_data6.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(filtered_data)
    
#days = 0
#interval = 7
#while days <= 1:
#    crawler()
#    time.sleep(interval)
#    days = days + 1 

def update_date_format(date):
    month_mapping = {
        "Jan": "January",
        "Feb": "February",
        "Mar": "March",
        "Apr": "April",
        "May": "May",
        "Jun": "June",
        "Jul": "July",
        "Aug": "August",
        "Sep": "September",
        "Oct": "October",
        "Nov": "November",
        "Dec": "December"}
    for month_abbr, month_full in month_mapping.items():
            if month_abbr in date:
                date = date.replace(month_abbr, month_full)
    return date

punc = set(string.punctuation)
filename = "important_data2.csv"
documents = []
sw = stopwords.words('english')
with open(filename, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        authors = [author.strip() for author in row[0].split(',')]
        title = row[2]  
        date = row[4]
        documents.append((authors, title, date))
        
        
#tokenization, removal of punctuation,stopwords, stemming
ps = PorterStemmer()
ready_docs = []
for doc in documents:
    authors, title, date = doc
    date = update_date_format(date)

    ready_authors = []
    for author in authors:
        author_tokens = word_tokenize(author.lower()) 
        ready_author = " ".join([ps.stem(w) for w in author_tokens if w not in punc and w not in sw])
        ready_authors.append(ready_author)

    title_tokens = word_tokenize(title.lower())  
    ready_title = " ".join([ps.stem(w) for w in title_tokens if w not in punc and w not in sw])
    ready_date = word_tokenize(date.lower())
    
    ready_doc = (ready_authors, ready_title, ready_date)
    ready_docs.append(ready_doc)


index = defaultdict(list)
for post_list, (authors, title, date) in enumerate(ready_docs, start=1):
    tokens = title.split() 
    for author in authors:
        tokens.extend(author.split()) 
    tokens.extend(date)
    for token in tokens:
        index[token].append(post_list)
        
stemmer = PorterStemmer()
def preprocess_query(query):
    query_terms = query.lower().split()
    stemmed_query = [stemmer.stem(term) for term in query_terms]
    return ' '.join(stemmed_query)

search_term = input("Enter your query here: ")
search_term = preprocess_query(search_term)
results = []  
search_results = []

def calculate_idf(term, index, total_documents):
    doc_count_with_term = len(index.get(term, []))
    return math.log(total_documents / (doc_count_with_term + 1))

# Function to calculate TF-IDF scores for the search query and documents
def calculate_tfidf_scores(query, index, documents):
    tfidf_scores = {}
    query_terms = query.lower().split()
    total_documents = len(documents)
    for term in query_terms:
        tf = query_terms.count(term)
        idf = calculate_idf(term, index, total_documents)
        tfidf_scores[term] = tf * idf
    return tfidf_scores

tfidf_scores = calculate_tfidf_scores(search_term, index, documents)

# Function to calculate the relevance score for a document based on TF-IDF scores
def calculate_relevance_score(doc_id, tfidf_scores, index):
    score = 0.0
    for term, tfidf_score in tfidf_scores.items():
        if doc_id in index.get(term, []):
            score += tfidf_score
    return round(score, 2) 

for doc_id in set(doc_id for term in search_term.lower().split() for doc_id in index.get(term, [])):
    authors, title, date = documents[doc_id - 1]
    result_id = (tuple(authors), tuple(title), date)
    relevance_score = calculate_relevance_score(doc_id, tfidf_scores, index)
    search_results.append((doc_id, relevance_score))
    results.append((authors, title, date))
                         
ranked_results = sorted(search_results, key=lambda x: x[1], reverse=True)   
def create_dataframe(results):
    df = pd.DataFrame(results, columns=[
        "Relevance Score",
        "Title",
        "Authors",
        "Date",
        "Title URL",
        "Authors URL"
    ])
    return df

def create_dataframe_html(results):
    df = pd.DataFrame(results, columns=[
        "Relevance Score",
        "Title",
        "Authors",
        "Date",
        "Title URL",
        "Authors URL"
    ])
    df['Title URL'] = df['Title URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
    df['Authors URL'] = df['Authors URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

    return df.to_html(escape=False)
urls_dict = {}
authors_urls_dict = {}
with open(filename, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        title = row[2]
        title_url = row[3]
        authors = tuple(row[0].split(", "))  
        authors_url = row[1] 
        urls_dict[title] = title_url
        authors_urls_dict[authors] = authors_url
        
for result in results:
    authors = tuple(result[0]) 
    authors_list = ', '.join(authors)
    title = result[1]
    date = result[2]
    title_url = urls_dict.get(title, "URL not available")  
    authors_url = authors_urls_dict.get(authors, "URL not available")  

    for doc_id in ranked_results:
        search_result = (
            relevance_score,
            title,
            authors_list,
            date,
            title_url,
            authors_url
        )
        search_results.append(search_result)   
search_results_df = create_dataframe(search_results)
search_results_df.dropna(subset=["Title URL", "Authors URL"], inplace=True)
search_results_df.drop_duplicates(subset=["Title"], keep="first", inplace=True)
search_results_df.reset_index(drop=True, inplace=True)

def make_clickable(urls):
    clickable_links = []
    for url in urls.split(", "):
        clickable_links.append(f'<a href="{url}" target="_blank">{url}</a>')
    return ', '.join(clickable_links)

search_results_df["Title URL"] = search_results_df["Title URL"].apply(make_clickable)
search_results_df["Authors URL"] = search_results_df["Authors URL"].apply(make_clickable)
search_results_df_html = search_results_df.to_html(escape=False, index=False)
display(HTML(search_results_df_html))

@app.route('/', methods=['GET', 'POST'])
def search_page():
    if request.method == 'POST':
        search_query = request.form.get('search_query', '').strip()
        if search_query:
            search_term = preprocess_query(search_query)
            results = []  
            search_results = []

            tfidf_scores = calculate_tfidf_scores(search_term, index, documents)

            for doc_id in set(doc_id for term in search_term.lower().split() for doc_id in index.get(term, [])):
                authors, title, date = documents[doc_id - 1]
                result_id = (tuple(authors), tuple(title), date)
                relevance_score = calculate_relevance_score(doc_id, tfidf_scores, index)
                search_results.append((doc_id, relevance_score))
                results.append((authors, title, date))

                if not search_results:
                    no_results_message = "No results found for the search query."
                else:
                    no_results_message = None

            ranked_results = sorted(search_results, key=lambda x: x[1], reverse=True)   

            def create_dataframe(results):
                df = pd.DataFrame(results, columns=[
                    "Title",
                    "Authors",
                    "Date",
                    "Title URL",
                    "Authors URL"
                ])
                return df

            def create_dataframe_html(results):
                df = pd.DataFrame(results, columns=[
                    "Title",
                    "Authors",
                    "Date",
                    "Title URL",
                    "Authors URL"
                ])
                df['Title URL'] = df['Title URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
                df['Authors URL'] = df['Authors URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

            urls_dict = {}
            authors_urls_dict = {}
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                headers = next(reader)
                for row in reader:
                    title = row[2]
                    title_url = row[3]
                    authors = tuple(row[0].split(", "))  
                    authors_url = row[1] 
                    urls_dict[title] = title_url
                    authors_urls_dict[authors] = authors_url
                    
            for result in results:
                authors = tuple(result[0]) 
                authors_list = ', '.join(authors)
                title = result[1]
                date = result[2]
                title_url = urls_dict.get(title, "URL not available")  
                authors_url = authors_urls_dict.get(authors, "URL not available")  

                for doc_id in ranked_results:
                    search_result = (
                        title,
                        authors_list,
                        date,
                        title_url,
                        authors_url
                    )
                    search_results.append(search_result)  
                 
            search_results_df = create_dataframe(search_results)
            search_results_df.dropna(subset=["Title URL", "Authors URL"], inplace=True)
            search_results_df.drop_duplicates(subset=["Title"], keep="first", inplace=True)
            search_results_df.reset_index(drop=True, inplace=True)

            def make_clickable(urls):
                clickable_links = []
                for url in urls.split(", "):
                    clickable_links.append(f'<a href="{url}" target="_blank">{url}</a>')
                return ', '.join(clickable_links)

            search_results_df["Title URL"] = search_results_df["Title URL"].apply(make_clickable)
            search_results_df["Authors URL"] = search_results_df["Authors URL"].apply(make_clickable)
            search_results_df_html = search_results_df.to_html(escape=False, index=False)
            return render_template('results.html', search_query=search_query, search_results=search_results_df_html)

    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)