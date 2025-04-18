import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import plotly.express as px
from textblob import TextBlob
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat Analytics API")

class FileUpload(BaseModel):
    filename: str
    file_content: str  # base64 encoded content

# Add this after the existing FileUpload class
class AnalyticsRequest(BaseModel):
    filename: str
    file_content: str  # base64 encoded content
    n_frequent_words: int = 5
    n_topics: int = 5
    n_clusters: int = 5
    start_date: str
    end_date: str

# Global variable to store Analytics instance
chat_analyzer = None

@app.post("/upload")
async def upload_file(file_data: FileUpload):
    try:
        # Decode base64 content
        content = base64.b64decode(file_data.file_content).decode('utf-8')
        
        # Save to temporary file
        temp_file = "temp_chat.txt"
        with open(temp_file, "w", encoding='utf-8') as f:
            f.write(content)
        
        # Initialize Analytics
        global chat_analyzer
        chat_analyzer = Analytics(temp_file)
        
        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/frequent-words")
async def get_frequent_words(n: int = 5):
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        image_data = chat_analyzer.get_top_frequent_words(n)
        return {
            "message": "Analysis completed",
            "visualization": image_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/message-length")
async def get_message_length():
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        result = chat_analyzer.message_length_analysis()
        return {
            "message": "Analysis completed",
            "data": result["stats"],
            "visualization": result["visualization"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-time")
async def get_active_time():
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        image_data = chat_analyzer.active_time_analysis()
        return {
            "message": "Analysis completed",
            "visualization": image_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-interaction")
async def get_user_interaction():
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        image_data = chat_analyzer.user_interaction()
        return {
            "message": "Analysis completed",
            "visualization": image_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topic-modeling")
async def get_topic_modeling(n_topics: int = 5):
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        result = chat_analyzer.topic_modeling(n_topics)
        return {
            "message": "Analysis completed",
            "topics": result["topics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-conversations")
async def get_similar_conversations(n_clusters: int = 5):
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        image_data = chat_analyzer.similar_conversations(n_clusters)
        return {
            "message": "Analysis completed",
            "visualization": image_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment")
async def get_sentiment():
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        image_data = chat_analyzer.sentiment_dashboard()
        return {
            "message": "Analysis completed",
            "visualization": image_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-summary")
async def get_conversation_summary(start_date: str, end_date: str):
    if not chat_analyzer:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    try:
        summary = chat_analyzer.get_conversation_summary(start_date, end_date)
        if "error" in summary:
            raise HTTPException(status_code=400, detail=summary["error"])
        return {
            "message": "Analysis completed",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this after the other endpoints
@app.post("/all")
async def get_all_analytics(request: AnalyticsRequest):
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)
    try:
        # First process the file upload
        content = base64.b64decode(request.file_content).decode('utf-8')
        
        # Save to temporary file
        temp_file = "temp_chat.txt"
        with open(temp_file, "w", encoding='utf-8') as f:    
            f.write(content)
        
        # Initialize Analytics
        global chat_analyzer
        chat_analyzer = Analytics(temp_file)

        # Now process all analytics
        results = {
            "frequent_words": {
                "visualization": chat_analyzer.get_top_frequent_words(request.n_frequent_words)
            },
            "message_length": chat_analyzer.message_length_analysis(),
            "active_time": {
                "visualization": chat_analyzer.active_time_analysis()
            },
            "user_interaction": {
                "visualization": chat_analyzer.user_interaction()
            },
            "topic_modeling": chat_analyzer.topic_modeling(request.n_topics),
            "similar_conversations": {
                "visualization": chat_analyzer.similar_conversations(request.n_clusters)
            },
            "sentiment": {
                "visualization": chat_analyzer.sentiment_dashboard()
            },
            "conversation_summary": chat_analyzer.get_conversation_summary(
                request.start_date, 
                request.end_date
            )
        }
        
        return {
            "message": "File processed and all analyses completed successfully",
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Analytics:
    def __init__(self, file):
        self.data = None
        self.file = file
        self.df = None
        self.users_messages = None
        self.users_most_frequent_words = {}
        self.setup()
        
    def setup(self):
        try:
            with open(self.file) as f:
                self.data = f.read()
        except FileNotFoundError:
            print("File not found")
            exit(1)
        except PermissionError:
            print("check your file permissions.")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            exit(1)             
        nltk.download('stopwords')
        combined_data = self.combine_multiline_messages(self.data)
        self.df = pd.DataFrame(combined_data)
        self.df["date"] = self.df[0].str.split(" ", expand=True)[0]
        self.df['date'] = pd.to_datetime(self.df['date'], dayfirst=True)
        self.df["time"] = self.df[0].str.split(" ", expand=True)[1]
        self.df["content"] = self.df[0].str.partition(",", expand=True)[2].str.partition("-", expand=True)[2]
        # Only split when ':' is present
        self.df['user'] = self.df['content'].apply(lambda x: x.split(":")[0] if ":" in x else "system")
        self.df['message'] = self.df['content'].apply(lambda x: x.split(":", 1)[1].strip() if ":" in x else x.strip())

        self.df = self.df.drop(columns=[0, "content"])
        self.df['message_length'] = self.df['message'].str.len()
        self.users_messages = self.get_user_message(self.df)
        
    def combine_multiline_messages(self, text):
        combined_lines = []
        pattern = re.compile(r"^\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?(am|pm)?\s?-\s")
        for line in text.splitlines()[1:]:
            if pattern.match(line):
                if " <Media omitted>" not in line:
                    combined_lines.append(line)
            else:
                if combined_lines:
                    combined_lines[-1] += ' ' + line.strip()
        return combined_lines
    
    def get_user_message(self, df):
        users_text = defaultdict(str)
        for index, row in df.iterrows():
            user = str(row["user"]).lstrip(" ")
            message = row["message"]
            users_text[user] += message
        return users_text
        
    def get_top_frequent_words(self, n=5):
        for user, text in self.users_messages.items():
            text = text.lower()
            words = re.findall(r'\b\w{4,}\b', text)  # only words with length >= 4
            word_counter = Counter(words)
            self.users_most_frequent_words[user] = word_counter.most_common(n)

        # Visualization
        data = []
        for user, word_freqs in self.users_most_frequent_words.items():
            for word, freq in word_freqs:
                data.append((user, word, freq))

        df_words = pd.DataFrame(data, columns=['User', 'Word', 'Frequency'])

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_words, x='Frequency', y='Word', hue='User', dodge=True)
        plt.title("Top Frequent Words Per User (Min 4 Characters)")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.legend(title='User', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def message_length_analysis(self):
        user_avg = self.df.groupby('user')['message_length'].mean().sort_values(ascending=False)
        daily_avg = self.df.groupby(self.df['date'].dt.date)['message_length'].mean()

        plt.figure(figsize=(10, 4))
        daily_avg.plot()
        plt.title("Average Message Length Over Time")
        plt.xlabel("Date")
        plt.ylabel("Avg Message Length")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        return {
            "stats": user_avg.to_dict(),
            "visualization": base64.b64encode(buf.getvalue()).decode('utf-8')
        }

    def active_time_analysis(self):
        # Clean up the time column if needed
        if isinstance(self.df['time'].iloc[0], str):
            self.df['time'] = self.df['time'].str.replace('\u202f', '', regex=True).str.strip()
        
        # Extract hour from the existing datetime 'date' column
        self.df['hour'] = self.df['date'].dt.hour
        # Extract weekday from the existing datetime 'date' column
        self.df['weekday'] = self.df['date'].dt.day_name()

        # Weekday activity
        plt.figure(figsize=(10, 4))
        sns.countplot(x='weekday', data=self.df, order=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        plt.title("Most Active Weekdays")
        plt.tight_layout()
        
        # Save to buffer instead of showing
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Return base64 encoded image
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def user_interaction(self):
        user_counts = Counter(self.df['user'])

        G = nx.Graph()
        for user, count in user_counts.items():
            G.add_node(user, size=count)

        users = list(user_counts.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                G.add_edge(users[i], users[j])

        sizes = [G.nodes[u]['size'] * 50 for u in G.nodes]

        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)  # consistent layout
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

        plt.title("User Interaction Graph", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        
        # Save to buffer instead of showing
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Return base64 encoded image
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def topic_modeling(self,n_topics=5):
        # Sample: df["message"]
        messages = self.df["message"]
        # 1. Convert messages to bag-of-words
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(messages)

        # 2. Apply LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 2 topics for example
        lda.fit(X)

        # 3. Get top words per topic
        words = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-6:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        return {"topics": topics}
    
    def similar_conversations(self,n_clusters=5):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        self.df['cleaned_message'] = self.df['message'].apply(
        lambda text: ' '.join(
        [word for word in re.sub(r'[^\w\s]', '', text.lower()).split() if word not in stop_words and len(word) > 3]
        )
    )


        # Generate sentence embeddings
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(self.df['cleaned_message'], show_progress_bar=False)


        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.df['cluster'] = kmeans.fit_predict(embeddings)

        # Dimensionality reduction + scaling
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        self.df[['x', 'y']] = MinMaxScaler().fit_transform(reduced)

        # Assign top keyword per cluster
        top_keywords = {}
        for c in self.df['cluster'].unique():
            words = ' '.join(self.df[self.df['cluster'] == c]['cleaned_message']).split()
            if words:
                top_words = [word for word, _ in Counter(words).most_common(10)]
                cluster_label = ', '.join(top_words).title()
            else:
                cluster_label = f'Cluster {c}'
            top_keywords[c] = cluster_label

        self.df['cluster_label'] = self.df['cluster'].map(top_keywords)

        # Plot
        fig = px.scatter(
            self.df,
            x='x',
            y='y',
            color='cluster_label',
            hover_data=['user', 'message'],
            title='Semantic Clustering of Messages',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.update_layout(legend_title_text='Cluster Topics')
        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def sentiment_dashboard(self):
        self.df['sentiment'] = self.df['message'].astype(str).apply(lambda x: float(TextBlob(x).sentiment.polarity))        
        # Sentiment per user
        user_sentiment = self.df.groupby('user')['sentiment'].mean().sort_values()
        
        plt.figure(figsize=(10, 4))
        user_sentiment.plot(kind='barh', color='skyblue')
        plt.title("Average Sentiment per User")
        plt.xlabel("Avg Sentiment Score")
        plt.tight_layout()
        
        # Save to buffer instead of showing
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Return base64 encoded image
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_conversation_summary(self, start_date, end_date):
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Filter messages between dates
        mask = (self.df['date'].dt.date >= start_date.date()) & (self.df['date'].dt.date <= end_date.date())
        filtered_df = self.df[mask]
        
        if filtered_df.empty:
            return {"error": "No messages found in the specified date range"}
        
        # Prepare conversation text for Gemini
        conversations = []
        for _, row in filtered_df.iterrows():
            conversations.append(f"{row['user']}: {row['message']}")
        
        conversation_text = "\n".join(conversations)
        
        # Setup Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Define a more focused task for concise summary
        task = """Provide a brief, one-paragraph summary of these chat conversations, focusing on the main themes and key interactions. 
               Include only the most significant topics and notable points discussed."""
        sys_instruct = """Provide the output in JSON format with a single key 'summary' containing a concise paragraph.
                       Keep the summary under 150 words."""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=task + sys_instruct,
                    response_mime_type='application/json',
                    temperature=0.3  # Lower temperature for more focused output
                ),
                contents=conversation_text
            )
            
            # Parse JSON response
            summary = json.loads(response.text)
            return summary
        
        except Exception as e:
            return {"error": f"Failed to generate summary: {str(e)}"}
