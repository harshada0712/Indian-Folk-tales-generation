import pandas as pd
import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK resources already downloaded or could not be downloaded")

# Function to create visualization directory
def create_vis_dir():
    """Create a directory for saving visualizations"""
    vis_dir = "visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    return vis_dir



# Step 2: Data visualization utilities
def visualize_dataset_stats(df, content_column, vis_dir):
    """
    Create and save visualization of dataset statistics
    """
    print("\n" + "="*80)
    print("📊 CREATING DATASET VISUALIZATIONS")
    print("="*80)

    # Create figure for word count distribution
    plt.figure(figsize=(12, 6))

    # Calculate word counts
    word_counts = df[content_column].apply(lambda x: len(str(x).split()))

    # Plot histogram of word counts
    sns.histplot(word_counts, kde=True)
    plt.title('Distribution of Word Counts in Stories', fontsize=15)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{vis_dir}/word_count_distribution.png", bbox_inches='tight')
    plt.close()
    print("✅ Created word count distribution visualization")

    # Create plot for story length distribution
    plt.figure(figsize=(10, 6))
    char_lengths = df[content_column].apply(len)
    sns.histplot(char_lengths, kde=True, color='green')
    plt.title('Distribution of Character Lengths in Stories', fontsize=15)
    plt.xlabel('Character Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{vis_dir}/char_length_distribution.png", bbox_inches='tight')
    plt.close()
    print("✅ Created character length distribution visualization")

    # Create sentence count distribution
    plt.figure(figsize=(10, 6))
    sentence_counts = df[content_column].apply(lambda x: len(sent_tokenize(str(x))))
    sns.histplot(sentence_counts, kde=True, color='purple')
    plt.title('Distribution of Sentence Counts in Stories', fontsize=15)
    plt.xlabel('Sentence Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{vis_dir}/sentence_count_distribution.png", bbox_inches='tight')
    plt.close()
    print("✅ Created sentence count distribution visualization")

    # Create wordcloud
    create_wordcloud(df, content_column, vis_dir)

    # Create top words visualization
    create_top_words_viz(df, content_column, vis_dir)


def create_wordcloud(df, content_column, vis_dir):
    """Create and save wordcloud visualization"""
    print("\n📝 Generating Word Cloud...")

    # Combine all text
    text = ' '.join(df[content_column].astype(str))

    # Create stopwords set
    stopwords_set = set(STOPWORDS)
    try:
        stopwords_set.update(set(stopwords.words('english')))
    except:
        pass

    # Generate wordcloud
    plt.figure(figsize=(12, 12))
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color='white',
        stopwords=stopwords_set,
        min_font_size=10,
        max_font_size=150,
        colormap='viridis',
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{vis_dir}/wordcloud.png", bbox_inches='tight')
    plt.close()
    print("✅ Created word cloud visualization")

def create_top_words_viz(df, content_column, vis_dir, top_n=30):
    """Create visualization of top words"""
    print("\n📊 Analyzing Most Common Words...")

    # Combine all text
    text = ' '.join(df[content_column].astype(str))

    # Get all words
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words and len(word) > 2]
    except:
        # If NLTK stopwords are not available
        words = [word for word in words if len(word) > 2]

    # Count word frequencies
    word_counts = Counter(words)

    # Get top words
    top_words = word_counts.most_common(top_n)

    # Extract words and frequencies
    labels = [word for word, count in top_words]
    values = [count for word, count in top_words]

    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    plt.barh(labels[::-1], values[::-1], color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
    plt.title(f'Top {top_n} Most Common Words', fontsize=15)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/top_words.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Created visualization of top {top_n} words")


    def visualize_story_samples(df, content_column, vis_dir, num_samples=5):
        """Display and visualize story samples"""
        print("\n📝 Displaying Story Samples...")

        # Get random samples
        samples = df.sample(min(num_samples, len(df)))

        # Create text file with samples
        with open(f"{vis_dir}/story_samples.txt", "w", encoding="utf-8") as f:
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                story = row[content_column]
                f.write(f"SAMPLE STORY #{i}\n")
                f.write("="*50 + "\n")
                f.write(story + "\n\n")

                # Print truncated version to console
                print(f"Sample #{i}: {story[:150]}..." if len(story) > 150 else story)

        print(f"✅ Saved {num_samples} sample stories to {vis_dir}/story_samples.txt")

def visualize_sentiment_distribution(df, content_column, vis_dir):
    """Visualize sentiment distribution in stories"""
    try:
        from textblob import TextBlob
        print("\n💭 Analyzing Sentiment Distribution...")

        # Function to get sentiment
        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity

        # Calculate sentiment for each story
        df['sentiment'] = df[content_column].apply(get_sentiment)

        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['sentiment'], kde=True, color='teal')
        plt.title('Sentiment Distribution in Stories', fontsize=15)
        plt.xlabel('Sentiment (Negative → Positive)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.savefig(f"{vis_dir}/sentiment_distribution.png", bbox_inches='tight')
        plt.close()
        print("✅ Created sentiment distribution visualization")

    except:
        print("⚠️ TextBlob not available. Skipping sentiment analysis.")


# Step 3: Load and clean the dataset with comprehensive data exploration
def load_and_clean_data(file_path):
    """
    Load and clean the dataset, with improved handling for various file formats,
    corrupt data, and comprehensive data exploration
    """
    print("\n" + "="*80)
    print("🔍 LOADING AND EXPLORING DATASET")
    print("="*80)

    # Check file extension
    _, ext = os.path.splitext(file_path)

    if ext.lower() == '.csv':
        # Try different approaches to load CSV file
        try:
            # First attempt: standard read
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded CSV file with standard parser")
        except Exception as e:
            print(f"Standard CSV parsing failed: {str(e)}")
            try:
                # Second attempt: with error handling
                df = pd.read_csv(file_path, on_bad_lines='skip')
                print(f"✅ Loaded CSV file with on_bad_lines='skip'")
            except Exception:
                try:
                    # Third attempt: with Python engine
                    df = pd.read_csv(file_path, engine='python')
                    print(f"✅ Loaded CSV file with Python engine")
                except Exception:
                    try:
                        # Fourth attempt: Read as text and parse manually
                        print("Trying to read file as text and parse manually...")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # Find header line
                        header = lines[0].strip().split(',')

                        # Process rows
                        data = []
                        for line in lines[1:]:
                            # Skip empty lines
                            if not line.strip():
                                continue

                            # Simple parsing - this won't handle all CSV edge cases
                            # but should work for most cases with simple text data
                            row = []
                            in_quotes = False
                            current_field = ""

                            for char in line:
                                if char == '"':
                                    in_quotes = not in_quotes
                                elif char == ',' and not in_quotes:
                                    row.append(current_field)
                                    current_field = ""
                                else:
                                    current_field += char

                            # Add the last field
                            row.append(current_field.strip())

                            # Ensure row has same length as header
                            while len(row) < len(header):
                                row.append("")

                            # Trim if too long
                            if len(row) > len(header):
                                row = row[:len(header)]

                            data.append(row)

                        # Create DataFrame
                        df = pd.DataFrame(data, columns=header)
                        print(f"✅ Created DataFrame manually from text content")
                    except Exception as e:
                        # Last resort: create a simple dataset from text content
                        print(f"All CSV parsing methods failed, creating simple dataset from file content: {str(e)}")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()

                            # Split text into paragraphs
                            paragraphs = re.split(r'\n\s*\n', text)
                            paragraphs = [p.strip() for p in paragraphs if p.strip()]

                            # Create DataFrame
                            df = pd.DataFrame({"Story Content": paragraphs})
                        except Exception as e:
                            raise ValueError(f"Failed to load CSV file: {str(e)}")
    elif ext.lower() == '.txt':
        # For text files, read as plain text and create a DataFrame
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split text into paragraphs or stories based on double newlines
            stories = re.split(r'\n\s*\n', text)
            stories = [s.strip() for s in stories if s.strip()]

            # Create DataFrame
            df = pd.DataFrame({"Story Content": stories})
            print(f"✅ Created DataFrame from text file with {len(stories)} stories/paragraphs")
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    stories = re.split(r'\n\s*\n', text)
                    stories = [s.strip() for s in stories if s.strip()]
                    df = pd.DataFrame({"Story Content": stories})
                    print(f"✅ Created DataFrame from text file with {encoding} encoding")
                    break
                except:
                    continue
            else:
                raise ValueError("Could not read text file with any encoding")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Display dataset info
    print("\n📊 DATASET OVERVIEW")
    print(f"• Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"• Columns: {', '.join(df.columns.tolist())}")

    # Print column information
    print("\n📋 COLUMN DETAILS:")
    for col in df.columns:
        print(f"• {col}: {df[col].dtype} - {df[col].nunique()} unique values - {df[col].isnull().sum()} missing values")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Identify content column
    content_column = identify_content_column(df)
    print(f"\n✅ Using '{content_column}' as the content column")

    # Clean text in content column
    print("\n🧹 Cleaning text data...")
    df[content_column] = df[content_column].apply(clean_text)
    print("✅ Text cleaning complete")

    # Display statistics of cleaned data
    print("\n📊 CLEANED DATA STATISTICS:")

    # Word count statistics
    word_counts = df[content_column].apply(lambda x: len(str(x).split()))
    print(f"• Average Word Count: {word_counts.mean():.1f} words")
    print(f"• Median Word Count: {word_counts.median():.1f} words")
    print(f"• Min Word Count: {word_counts.min()} words")
    print(f"• Max Word Count: {word_counts.max()} words")

    # Character count statistics
    char_counts = df[content_column].apply(len)
    print(f"• Average Character Count: {char_counts.mean():.1f} characters")

    # Sentence count statistics
    sentence_counts = df[content_column].apply(lambda x: len(sent_tokenize(str(x))))
    print(f"• Average Sentence Count: {sentence_counts.mean():.1f} sentences")

    # Remove empty values
    df = df.dropna(subset=[content_column])
    print(f"\n✅ Removed empty values. Final dataset size: {df.shape[0]} rows")

    # Display visualizations - using created visualization directory
    vis_dir = create_vis_dir()
    visualize_dataset_stats(df, content_column, vis_dir)
    visualize_story_samples(df, content_column, vis_dir)

    try:
        visualize_sentiment_distribution(df, content_column, vis_dir)
    except:
        print("⚠️ Skipping sentiment visualization (required libraries not available)")

    # Save cleaned dataset
    cleaned_path = "cleaned_stories_fixed.csv"
    df.to_csv(cleaned_path, index=False, quoting=1)  # quoting=1 for proper escaping
    print(f"\n✅ Dataset cleaned and saved to {cleaned_path}")

    return df, content_column


def identify_content_column(df):
    """
    Identify the column containing story content
    """
    # First, try exact matches
    for name in ["Story Content", "story content", "Content", "content", "Text", "text"]:
        if name in df.columns:
            return name

    # Next, try partial matches
    for col in df.columns:
        if "content" in col.lower() or "story" in col.lower() or "text" in col.lower():
            return col

    # If nothing found, use the column with the longest text on average
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    if text_cols:
        avg_lengths = {col: df[col].astype(str).apply(len).mean() for col in text_cols}
        return max(avg_lengths, key=avg_lengths.get)

    # Last resort: first column
    return df.columns[0]

def clean_text(text):
    """
    Clean and normalize text data
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special formatting but keep important punctuation
    text = re.sub(r'[^\w\s.,!?:;()\'\"-]', '', text)

    return text

# Step 4: Create datasets for training and evaluation
def prepare_datasets(df, content_column, test_size=0.1):
    """
    Prepare train and evaluation datasets with visualization
    """
    print("\n" + "="*80)
    print("🔢 PREPARING TRAINING AND EVALUATION DATASETS")
    print("="*80)

    # Extract stories
    stories = df[content_column].dropna().astype(str).tolist()

    # Filter out very short stories (likely noise)
    original_count = len(stories)
    stories = [story for story in stories if len(story.split()) >= 20]
    filtered_count = len(stories)

    print(f"• Original story count: {original_count}")
    print(f"• Filtered out {original_count - filtered_count} very short stories (< 20 words)")
    print(f"• Final story count for model training: {filtered_count}")

    # Split into train and eval sets
    train_texts, eval_texts = train_test_split(stories, test_size=test_size, random_state=42)

    print(f"\n✅ Dataset split: {len(train_texts)} training stories and {len(eval_texts)} evaluation stories")

    # Visualize the split
    vis_dir = create_vis_dir()
    plt.figure(figsize=(8, 6))
    plt.pie([len(train_texts), len(eval_texts)],
            labels=['Training Set', 'Evaluation Set'],
            autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'],
            explode=[0, 0.1],
            startangle=90,
            shadow=True)
    plt.title('Dataset Split for Training and Evaluation', fontsize=15)
    plt.axis('equal')
    plt.savefig(f"{vis_dir}/dataset_split.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Created visualization of dataset split")

    # Visualize length distributions in train/eval sets
    plt.figure(figsize=(12, 6))
    train_lengths = [len(text.split()) for text in train_texts]
    eval_lengths = [len(text.split()) for text in eval_texts]

    plt.hist([train_lengths, eval_lengths], bins=30,
             label=['Training Set', 'Evaluation Set'],
             alpha=0.7, color=['#3498db', '#e74c3c'])
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Word Count Distribution in Training and Evaluation Sets', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{vis_dir}/train_eval_length_distribution.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Created visualization of length distributions in train/eval sets")

    # Create Dataset objects
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    return train_dataset, eval_dataset


def tokenize_datasets(train_dataset, eval_dataset, tokenizer, max_length=512):
    """
    Tokenize datasets for training with visualization
    """
    print("\n" + "="*80)
    print("🔤 TOKENIZING DATASETS")
    print("="*80)

    # Add special tokens for story beginning and end
    special_tokens = {
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>"
    }

    # Add special tokens to tokenizer
    tokenizer.add_special_tokens(special_tokens)
    print("✅ Added special tokens to tokenizer")
    print(f"• Special tokens: {', '.join(special_tokens.values())}")
    print(f"• Tokenizer vocabulary size: {len(tokenizer)}")

    # Tokenization function with story markers
    def tokenize_function(examples):
        # Add special tokens to mark beginning and end of stories
        texts = [f"{tokenizer.bos_token} {text} {tokenizer.eos_token}" for text in examples["text"]]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    # Apply tokenization
    print("\n⏳ Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    print("⏳ Tokenizing evaluation dataset...")
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    print("\n✅ Tokenization complete")
    print(f"• Number of training examples: {len(tokenized_train)}")
    print(f"• Number of evaluation examples: {len(tokenized_eval)}")

    # Visualize example tokenization
    vis_dir = create_vis_dir()
    plt.figure(figsize=(14, 6))

    # Fix for token counting - handling tensor vs list correctly
    try:
        # Get sample text and its tokenization
        sample_idx = 0

        # Calculate token counts excluding padding
        # First convert to numpy arrays for easier comparison
        token_counts = []
        for i in range(len(tokenized_train)):
            input_ids = tokenized_train[i]['input_ids'].numpy() if hasattr(tokenized_train[i]['input_ids'], 'numpy') else tokenized_train[i]['input_ids']
            # Count non-padding tokens
            count = sum(1 for token_id in input_ids if token_id != tokenizer.pad_token_id)
            token_counts.append(count)

        plt.hist(token_counts, bins=30, color='purple', alpha=0.7)
        plt.axvline(x=max_length, color='red', linestyle='--', label=f'Max Length ({max_length})')
        plt.xlabel('Number of Tokens (excluding padding)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Token Counts in Training Set', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"{vis_dir}/token_count_distribution.png", bbox_inches='tight')
        plt.close()
        print(f"✅ Created visualization of token count distribution")
    except Exception as e:
        print(f"⚠️ Error creating token count visualization: {str(e)}")
        print("Continuing with tokenized datasets...")

    return tokenized_train, tokenized_eval


from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, DataCollatorForLanguageModeling
import os
import time
import torch

def train_model(tokenized_train, tokenized_eval, tokenizer, output_dir="./fine_tuned_folktales_gpt2"):
    """
    Train the model with improved parameters and visualizations
    """
    print("\n" + "="*80)
    print("🧠 TRAINING THE MODEL")
    print("="*80)

    # Create log directory
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load the base model
    print("\n⏳ Loading base GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("✅ Base model loaded")

    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Resized token embeddings for special tokens")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    print("✅ Created data collator for language modeling")

    # Check transformers version to determine available parameters
    import transformers
    print(f"\n📚 Transformers version: {transformers.__version__}")

    # Get device information
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Training on: {device.upper()}")

    # Set number of epochs and learning rate
    num_epochs = 5  # Reduced for demonstration
    learning_rate = 3e-5

    # Try to create training arguments with version-appropriate parameters
    try:
        # Try with newer API that includes evaluation_strategy
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            weight_decay=0.01,
            num_train_epochs=num_epochs,
            warmup_steps=100,
            logging_dir=log_dir,
            logging_steps=20,  # Log more frequently for better visualization
            save_steps=100,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=100,  # Evaluate more frequently
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            report_to="none",  # Disable wandb/tensorboard reporting
        )
        print("✅ Using newer Transformers API with evaluation_strategy")
    except TypeError:
        # Fall back to older API without evaluation_strategy
        print("⚠️ Using older Transformers API (evaluation_strategy not supported)")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            weight_decay=0.01,
            num_train_epochs=num_epochs,
            warmup_steps=100,
            logging_dir=log_dir,
            logging_steps=20,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available()
        )

    # Define the trainer with version compatibility
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval
    )

    # Print training configuration
    print("\n🔧 TRAINING CONFIGURATION:")
    print(f"• Number of epochs: {num_epochs}")
    print(f"• Learning rate: {learning_rate}")
    print(f"• Train batch size: {training_args.per_device_train_batch_size}")
    print(f"• Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"• Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    # Train the model
    print("\n⏳ Starting model training...")
    start_time = time.time()

    # Train the model and capture metrics
    train_result = trainer.train()

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model fine-tuned and saved to {output_dir}")
    return model, tokenizer


    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')


    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    import os


    # Step 2: Load cleaned dataset
    file_path = "cleaned_stories.csv"
    try:
        df, content_column = load_and_clean_data(file_path)
        print("✅ Cleaned dataset loaded")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        raise

    # Step 3: Prepare train/eval datasets
    train_dataset, eval_dataset = prepare_datasets(df, content_column)

    # Step 4: Initialize tokenizer
    print("\n⏳ Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("✅ Tokenizer initialized")


    # Step 5: Tokenize datasets
    tokenized_train, tokenized_eval = tokenize_datasets(train_dataset, eval_dataset, tokenizer)


    # Step 6: Train the model
output_dir = "./fine_tuned_folktales_gpt2"
train_model(tokenized_train, tokenized_eval, tokenizer, output_dir)

# Step 7: Generate stories based on prompts
print("\n" + "="*80)
print("📝 GENERATING STORIES FROM FINE-TUNED MODEL USING PROMPTS")
print("="*80)

try:
    print("\n⏳ Loading fine-tuned model...")
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_tuned_model.to(device)
    print(f"✅ Model loaded on {device.upper()}")

    # Define prompts for story generation
    prompts = [
        "Once upon a time in a small village,",
        "In the ancient kingdom of dragons,",
        "The wise old wolf told the children,",
        "Deep in the enchanted forest,",
        "The magical artifacts were hidden in"
    ]

    max_length = 200
    temperature = 0.8

    print(f"\n⏳ Generating stories from {len(prompts)} different prompts...")

    vis_dir = create_vis_dir()
    with open(f"{vis_dir}/generated_stories_from_prompts.txt", "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            # Encode the prompt text for the model
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            output = fine_tuned_model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            f.write(f"PROMPT: {prompt}\n")
            f.write("="*50 + "\n")
            f.write(generated_text + "\n\n")
            
            print(f"\nPrompt #{i+1}: \"{prompt}\"")
            print(f"Generated: {generated_text[:150]}..." if len(generated_text) > 150 else generated_text)

    print(f"\n✅ Generated stories saved to {vis_dir}/generated_stories_from_prompts.txt")
    
    # Function to generate a story from a user-provided prompt
    def generate_story_from_prompt(user_prompt, max_length=200, temperature=0.8):
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
        output = fine_tuned_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Demonstrate custom prompt usage
    custom_prompt = "The magical crystal began to glow as"
    print("\n⏳ Testing custom prompt functionality...")
    custom_story = generate_story_from_prompt(custom_prompt)
    print(f"\nCustom Prompt: \"{custom_prompt}\"")
    print(f"Generated: {custom_story[:150]}..." if len(custom_story) > 150 else custom_story)
    
    # Save the custom story too
    with open(f"{vis_dir}/custom_prompt_story.txt", "w", encoding="utf-8") as f:
        f.write(f"CUSTOM PROMPT: {custom_prompt}\n")
        f.write("="*50 + "\n")
        f.write(custom_story)
        
    print(f"\n✅ Custom prompt story saved to {vis_dir}/custom_prompt_story.txt")

    if is_colab:
        print("\n📥 Downloading generated files...")
        try:
            os.system("zip -r folktales_visualizations.zip visualizations/")
            files.download('folktales_visualizations.zip')
            os.system("zip -r fine_tuned_folktales_model.zip fine_tuned_folktales_gpt2/")
            files.download('fine_tuned_folktales_model.zip')
            files.download('cleaned_stories.csv')
        except Exception as e:
            print(f"⚠️ Error downloading files: {str(e)}")

except Exception as e:
    print(f"❌ Error generating stories: {str(e)}")

    print("\n" + "="*80)
    print("✨ FOLKTALES GPT-2 FINE-TUNING PIPELINE COMPLETED")
    print("="*80)



