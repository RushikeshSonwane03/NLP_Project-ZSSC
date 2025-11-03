"""
Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts
ACL 2023 - Clean Implementation with UI
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import torch
import nltk
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pathlib import Path
import time

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class ZSSCPipeline:
    def __init__(self, model_name='bert-base-uncased', dataset='sst2'):
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cache_dir = Path("model_cache")  # Add this
        self.model_cache_dir.mkdir(exist_ok=True)  # Add this

    def load_model(self, progress_callback=None):
        """Load BERT model and tokenizer with caching"""
        # Create cache file paths
        model_cache_file = self.model_cache_dir / f"{self.model_name.replace('/', '_')}_model.pkl"
        tokenizer_cache_file = self.model_cache_dir / f"{self.model_name.replace('/', '_')}_tokenizer.pkl"
        
        # Try to load from cache first
        if model_cache_file.exists() and tokenizer_cache_file.exists():
            if progress_callback:
                progress_callback(f"Loading cached {self.model_name} from disk...")
            
            try:
                with open(tokenizer_cache_file, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                with open(model_cache_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.model.to(self.device)
                self.model.eval()
                
                if progress_callback:
                    progress_callback(f"✓ Loaded {self.model_name} from cache on {self.device}")
                return
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Cache loading failed: {e}. Loading from HuggingFace...")
        
        # Load from HuggingFace if cache doesn't exist or failed
        if progress_callback:
            progress_callback(f"Loading {self.model_name} from HuggingFace (first time)...")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Save to cache
        if progress_callback:
            progress_callback("Saving model to cache for future use...")
        
        try:
            with open(tokenizer_cache_file, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            
            # Move model to CPU before saving to reduce file size
            self.model.cpu()
            with open(model_cache_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Move back to device
            self.model.to(self.device)
            
            if progress_callback:
                progress_callback(f"✓ Model cached successfully. Next load will be faster!")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Could not cache model: {e}")
        
        if progress_callback:
            progress_callback(f"✓ Loaded {self.model_name} on {self.device}")

    # def load_model(self, progress_callback=None):
    #     """Load BERT model and tokenizer"""
    #     if progress_callback:
    #         progress_callback("Loading BERT model...")
        
    #     self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    #     self.model = BertForMaskedLM.from_pretrained(self.model_name)
    #     self.model.to(self.device)
    #     self.model.eval()
        
    #     if progress_callback:
    #         progress_callback(f"✓ Loaded {self.model_name} on {self.device}")
    
    def load_data(self, dataset_path, train_subset=None, test_subset=None, 
                  progress_callback=None):
        """Load dataset with optional subset sampling"""
        if progress_callback:
            progress_callback(f"Loading dataset from {dataset_path}...")
        
        train_path = Path(dataset_path) / "train.csv"
        test_path = Path(dataset_path) / "test.csv"
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Random sampling if subset size specified
        if train_subset and train_subset < len(train_data):
            train_data = train_data.sample(n=train_subset, random_state=42)
            if progress_callback:
                progress_callback(f"Sampled {train_subset} train samples randomly")
        
        if test_subset and test_subset < len(test_data):
            test_data = test_data.sample(n=test_subset, random_state=42)
            if progress_callback:
                progress_callback(f"Sampled {test_subset} test samples randomly")
        
        columns = list(train_data.columns)
        train_labels = list(train_data[columns[0]])
        train_sentences = list(train_data[columns[1]])
        
        test_labels = list(test_data[columns[0]])
        test_sentences = list(test_data[columns[1]])
        
        if progress_callback:
            progress_callback(f"✓ Loaded {len(train_sentences)} train, {len(test_sentences)} test samples")
        
        return {
            'train_labels': train_labels,
            'train_sentences': train_sentences,
            'test_labels': test_labels,
            'test_sentences': test_sentences,
            'all_sentences': train_sentences + test_sentences
        }
    
    def parse_input_template(self, input_template, input_mapping):
        """Parse base prompt template"""
        template = input_template.replace("<sentence>", "")
        template = template.replace(".", "").strip()
        mapping_keys = list(input_mapping.keys())
        template = template.replace("_", mapping_keys[0])
        return template
    
    def positional_feature(self, template):
        """Apply positioning technique"""
        return [
            f"<sentence> . {template} .",
            f"{template} . <sentence> ."
        ]
    
    def reasoning_feature(self, template):
        """Apply subordination technique"""
        return [
            f"<sentence> so {template} .",
            f"{template} because <sentence> ."
        ]
    
    def paraphrase_feature(self, template, paraphrasing_tokens):
        """Apply paraphrasing technique"""
        templates = []
        template_split = template.split(" ")
        
        for i in range(len(template_split) - 1):
            pos_tag = nltk.pos_tag([template_split[i]])[0][1]
            
            for token in paraphrasing_tokens[i]:
                token_pos_tag = nltk.pos_tag([token])[0][1]
                if token_pos_tag == pos_tag:
                    update_template = template_split.copy()
                    update_template[i] = token
                    templates.append(" ".join(update_template))
        
        return templates
    
    def generate_paraphrasing_tokens(self, template, sample_sentence, top_k=30):
        """Generate paraphrasing tokens using MLM"""
        template_split = template.split(" ")
        paraphrasing_tokens = []
        
        for i in range(len(template_split) - 1):
            update_template = template_split.copy()
            update_template[i] = "[MASK]"
            masked_template = " ".join(update_template) + " because " + sample_sentence
            
            inputs = self.tokenizer(masked_template, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            input_ids = inputs['input_ids'][0].tolist()
            mask_index = input_ids.index(self.tokenizer.mask_token_id)
            predictions = outputs.logits[0, mask_index].cpu()
            
            top_tokens = torch.topk(predictions, top_k).indices.tolist()
            pred_tokens = self.tokenizer.convert_ids_to_tokens(top_tokens)
            paraphrasing_tokens.append(pred_tokens)
        
        return paraphrasing_tokens
    
    def generate_templates(self, base_prompt, data, input_mapping, progress_callback=None):
        """Step 1: Generate candidate prompts"""
        if progress_callback:
            progress_callback("=" * 50)
            progress_callback("STEP 1: GENERATING TEMPLATES")
            progress_callback("=" * 50)
        
        parsed_template = self.parse_input_template(base_prompt, input_mapping)
        
        if progress_callback:
            progress_callback(f"Base prompt: {base_prompt}")
            progress_callback(f"Parsed template: {parsed_template}")
        
        # Find sample sentence with mapping token
        mapping_tokens = list(input_mapping.values())
        sample_sentence = None
        for sentence in data['all_sentences']:
            if mapping_tokens[0].lower() in sentence.lower():
                sample_sentence = sentence.lower()
                break
        
        if not sample_sentence:
            sample_sentence = data['all_sentences'][0].lower()
        
        if progress_callback:
            progress_callback(f"Sample sentence for paraphrasing: {sample_sentence[:100]}...")
        
        # Generate paraphrasing tokens
        if progress_callback:
            progress_callback("Generating paraphrasing tokens using MLM...")
        
        paraphrasing_tokens = self.generate_paraphrasing_tokens(
            parsed_template, sample_sentence
        )
        
        # Generate templates using all techniques
        if progress_callback:
            progress_callback("Applying augmentation techniques...")
        
        paraphrase_templates = self.paraphrase_feature(parsed_template, paraphrasing_tokens)
        
        templates = []
        for template in paraphrase_templates:
            templates.extend(self.positional_feature(template))
            templates.extend(self.reasoning_feature(template))
        
        # Remove duplicates
        templates = list(set(templates))
        
        if progress_callback:
            progress_callback(f"✓ Generated {len(templates)} unique candidate prompts")
            progress_callback(f"Sample templates: {templates[:3]}")
        
        return templates, paraphrasing_tokens
    
    def create_synonym_samples(self, data, input_mapping, synonyms_list, max_samples=None):
        """Create samples with synonym replacements for ranking"""
        mapping_tokens = list(input_mapping.values())
        sample_reviews = []
        sample_labels = []
        
        for sentence in data['all_sentences']:
            if max_samples and len(sample_reviews) >= max_samples:
                break
                
            sentence = sentence.lower()
            
            for j, token in enumerate(mapping_tokens):
                if token in sentence:
                    temp = []
                    temp_label = []
                    
                    # Original sentence
                    temp.append(sentence)
                    temp_label.append(0)
                    
                    # Replace with opposite mapping
                    opposite_idx = 1 - j
                    temp.append(sentence.replace(token, mapping_tokens[opposite_idx]))
                    temp_label.append(1)
                    
                    # Replace with synonyms of same polarity
                    for syn in synonyms_list[j]:
                        temp.append(sentence.replace(token, syn))
                        temp_label.append(0)
                    
                    # Replace with synonyms of opposite polarity
                    for syn in synonyms_list[opposite_idx]:
                        temp.append(sentence.replace(token, syn))
                        temp_label.append(1)
                    
                    if temp:
                        sample_reviews.append(temp)
                        sample_labels.append(temp_label)
        
        return sample_reviews, sample_labels
    
    def format_sentence(self, sentence, template):
        """Format sentence with template"""
        sentence = sentence.replace(".", "")
        formatted = template.replace("<sentence>", sentence)
        formatted = formatted.replace("  ", " ")
        formatted = formatted.lower()
        formatted = formatted.replace("positive", "[MASK]")
        return formatted
    
    def get_prediction(self, sentence, mapping_token_ids):
        """Get prediction for a sentence"""
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        input_ids = inputs['input_ids'][0].tolist()
        mask_index = input_ids.index(self.tokenizer.mask_token_id)
        predictions = outputs.logits[0, mask_index].cpu().numpy()
        
        scores = [predictions[idx] for idx in mapping_token_ids]
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        return probs, np.argmax(probs)
    
    def score_templates(self, templates, sample_reviews, sample_labels, 
                       input_mapping, progress_callback=None):
        """Step 2: Score and rank templates"""
        if progress_callback:
            progress_callback("\n" + "=" * 50)
            progress_callback("STEP 2: SCORING TEMPLATES")
            progress_callback("=" * 50)
        
        mapping_tokens = list(input_mapping.values())
        token_text = " ".join(mapping_tokens)
        token_ids = self.tokenizer(token_text, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        
        if progress_callback:
            progress_callback(f"Scoring {len(templates)} templates...")
            progress_callback(f"Using {len(sample_reviews)} synonym sample groups")
        
        template_scores = []
        
        for idx, template in enumerate(templates):
            if progress_callback and (idx % 10 == 0 or idx == len(templates) - 1):
                progress_callback(f"Processed {idx+1}/{len(templates)} templates...")
            
            score = 0
            
            for reviews, labels in zip(sample_reviews, sample_labels):
                predictions = []
                
                for review in reviews:
                    formatted = self.format_sentence(review, template)
                    _, pred_label = self.get_prediction(formatted, token_ids)
                    predictions.append(pred_label)
                
                # Score based on label flips
                zero_label = predictions[0]
                non_zero_label = 1 - zero_label
                
                for pred, label in zip(predictions, labels):
                    if label == 0 and pred == zero_label:
                        score += 1
                    elif label == 1 and pred == non_zero_label:
                        score += 1
            
            template_scores.append(score)
        
        # Sort templates by score
        sorted_pairs = sorted(zip(template_scores, templates), reverse=True)
        sorted_scores, sorted_templates = zip(*sorted_pairs)
        
        if progress_callback:
            progress_callback(f"✓ Template scoring complete")
            progress_callback(f"Top score: {sorted_scores[0]}, Lowest score: {sorted_scores[-1]}")
            progress_callback(f"\nTop 3 templates:")
            for i in range(min(3, len(sorted_templates))):
                progress_callback(f"  {i+1}. [{sorted_scores[i]}] {sorted_templates[i]}")
        
        return list(sorted_templates), list(sorted_scores)
    
    def evaluate_template(self, template, sentences, input_mapping):
        """Evaluate a single template"""
        mapping_tokens = list(input_mapping.values())
        token_text = " ".join(mapping_tokens)
        token_ids = self.tokenizer(token_text, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        
        predictions = []
        
        for sentence in sentences:
            formatted = self.format_sentence(sentence, template)
            _, pred_label = self.get_prediction(formatted, token_ids)
            predictions.append(pred_label)
        
        return predictions
    
    def evaluate_templates(self, sorted_templates, data, input_mapping, 
                          top_k=1, progress_callback=None):
        """Step 3: Evaluate top templates"""
        if progress_callback:
            progress_callback("\n" + "=" * 50)
            progress_callback("STEP 3: EVALUATING TEMPLATES")
            progress_callback("=" * 50)
        
        results = {}
        
        # Evaluate on test set
        if progress_callback:
            progress_callback(f"\nEvaluating top-{top_k} template(s) on test set...")
        
        if top_k == 1:
            template = sorted_templates[0]
            if progress_callback:
                progress_callback(f"Using template: {template}")
            
            test_preds = self.evaluate_template(
                template, data['test_sentences'], input_mapping
            )
            
            results['test_predictions'] = test_preds
            results['test_accuracy'] = accuracy_score(data['test_labels'], test_preds)
            results['test_f1'] = f1_score(data['test_labels'], test_preds, average='macro')
            
        else:
            # Majority voting for top-k
            all_preds = []
            for i in range(top_k):
                template = sorted_templates[i]
                preds = self.evaluate_template(
                    template, data['test_sentences'], input_mapping
                )
                all_preds.append(preds)
            
            # Aggregate predictions
            test_preds = []
            for i in range(len(data['test_sentences'])):
                votes = [preds[i] for preds in all_preds]
                test_preds.append(1 if sum(votes) > top_k / 2 else 0)
            
            results['test_predictions'] = test_preds
            results['test_accuracy'] = accuracy_score(data['test_labels'], test_preds)
            results['test_f1'] = f1_score(data['test_labels'], test_preds, average='macro')
        
        if progress_callback:
            progress_callback(f"✓ Test Accuracy: {results['test_accuracy']:.4f}")
            progress_callback(f"✓ Test F1 Score: {results['test_f1']:.4f}")
        
        return results


def main():
    st.set_page_config(page_title="ZS-SC: Zero-Shot Sentiment Classification", layout="wide")
    
    st.title("🎯 Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts")
    st.markdown("**ACL 2023** - Interactive Implementation")
    
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["sst2", "mr", "cr"],
        help="Select benchmark dataset"
    )
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Model",
        ["bert-base-uncased", "bert-large-uncased"],
        help="Select BERT model"
    )
    
    # Dataset path
    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        f"dataset/original/{dataset}",
        help="Path to dataset folder containing train.csv and test.csv"
    )
    
    # Subset size controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Dataset Subset Size")
    
    use_subset = st.sidebar.checkbox("Use subset (faster processing)", value=True)
    
    if use_subset:
        train_subset = st.sidebar.number_input(
            "Train subset size",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            help="Number of random train samples to use"
        )
        test_subset = st.sidebar.number_input(
            "Test subset size",
            min_value=10,
            max_value=2000,
            value=200,
            step=10,
            help="Number of random test samples to use"
        )
        max_synonym_samples = st.sidebar.number_input(
            "Max synonym samples for scoring",
            min_value=10,
            max_value=200,
            value=30,
            step=5,
            help="Limit synonym replacement samples to speed up scoring"
        )
    else:
        train_subset = None
        test_subset = None
        max_synonym_samples = None
        st.sidebar.warning("⚠️ Using full dataset may take considerable time")
    
    st.sidebar.markdown("---")
    
    # Base prompt
    base_prompt = st.sidebar.text_input(
        "Base Prompt",
        "<sentence>. The sentence was _",
        help="Base prompt template"
    )
    
    # Input mapping
    st.sidebar.subheader("Label Mapping")
    positive_word = st.sidebar.text_input("Positive →", "great")
    negative_word = st.sidebar.text_input("Negative →", "terrible")
    
    # Synonyms
    st.sidebar.subheader("Synonyms (comma-separated)")
    positive_synonyms = st.sidebar.text_input(
        "Positive synonyms",
        "excellent, outstanding, wonderful, fantastic"
    )
    negative_synonyms = st.sidebar.text_input(
        "Negative synonyms",
        "awful, horrible, poor, bad"
    )
    
    # Top-k evaluation
    top_k = st.sidebar.slider("Top-k for evaluation", 1, 5, 1)
    
    # Paper results for comparison
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Paper Results (Table 2)")
    
    paper_results = {
        'sst2': {'bert-base': {'acc': 0.7218, 'f1': 0.7236}, 
                 'bert-large': {'acc': 0.7474, 'f1': 0.7471}},
        'mr': {'bert-base': {'acc': 0.6824, 'f1': 0.6826}, 
               'bert-large': {'acc': 0.7029, 'f1': 0.7036}},
        'cr': {'bert-base': {'acc': 0.7509, 'f1': 0.7210}, 
               'bert-large': {'acc': 0.8047, 'f1': 0.7843}}
    }
    
    model_key = 'bert-base' if 'base' in model_name else 'bert-large'
    st.sidebar.markdown(f"**{dataset.upper()} - {model_key}:**")
    st.sidebar.markdown(f"- Accuracy: {paper_results[dataset][model_key]['acc']:.4f}")
    st.sidebar.markdown(f"- F1 Score: {paper_results[dataset][model_key]['f1']:.4f}")
    st.sidebar.caption("*Full dataset results")
    
    # Run button
    if st.sidebar.button("🚀 Run ZS-SC Pipeline", type="primary"):
        # Prepare configuration
        input_mapping = {'positive': positive_word, 'negative': negative_word}
        synonyms_list = [
            [s.strip() for s in positive_synonyms.split(',')],
            [s.strip() for s in negative_synonyms.split(',')]
        ]
        
        # Create progress display
        progress_container = st.container()
        log_container = st.empty()
        logs = []
        
        def log_message(msg):
            logs.append(msg)
            log_container.text_area("Progress Log", "\n".join(logs[-50:]), height=400)
            time.sleep(0.01)  # Allow UI to update
        
        with progress_container:
            st.header("Pipeline Execution")
            
            try:
                # Initialize pipeline
                pipeline = ZSSCPipeline(model_name=model_name, dataset=dataset)
                
                # Load model
                with st.spinner("Loading model..."):
                    pipeline.load_model(log_message)
                
                # Load data
                with st.spinner("Loading dataset..."):
                    data = pipeline.load_data(
                        dataset_path, 
                        train_subset=train_subset,
                        test_subset=test_subset,
                        progress_callback=log_message
                    )
                
                # Step 1: Generate templates
                with st.spinner("Generating templates..."):
                    templates, paraphrasing_tokens = pipeline.generate_templates(
                        base_prompt, data, input_mapping, log_message
                    )
                
                # Create synonym samples
                log_message("\nCreating synonym replacement samples for scoring...")
                sample_reviews, sample_labels = pipeline.create_synonym_samples(
                    data, input_mapping, synonyms_list, max_samples=max_synonym_samples
                )
                log_message(f"✓ Created {len(sample_reviews)} sample groups")
                
                # Step 2: Score templates
                with st.spinner("Scoring templates (this may take a while)..."):
                    sorted_templates, sorted_scores = pipeline.score_templates(
                        templates, sample_reviews, sample_labels, 
                        input_mapping, log_message
                    )
                
                # Step 3: Evaluate
                with st.spinner("Evaluating templates..."):
                    results = pipeline.evaluate_templates(
                        sorted_templates, data, input_mapping, 
                        top_k, log_message
                    )
                
                # Display results
                st.success("✅ Pipeline completed successfully!")
                
                st.header("📊 Final Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
                    paper_acc = paper_results[dataset][model_key]['acc']
                    diff_acc = results['test_accuracy'] - paper_acc
                    st.caption(f"Paper: {paper_acc:.4f} (Δ {diff_acc:+.4f})")
                
                with col2:
                    st.metric("Test F1 Score", f"{results['test_f1']:.4f}")
                    paper_f1 = paper_results[dataset][model_key]['f1']
                    diff_f1 = results['test_f1'] - paper_f1
                    st.caption(f"Paper: {paper_f1:.4f} (Δ {diff_f1:+.4f})")
                
                with col3:
                    st.metric("Test Samples", len(data['test_sentences']))
                    st.caption(f"Top-{top_k} template(s)")
                
                # Top templates
                st.subheader("🏆 Top-5 Templates")
                for i in range(min(10, len(sorted_templates))):
                    with st.expander(f"#{i+1} - Score: {sorted_scores[i]}"):
                        st.code(sorted_templates[i], language=None)
                
                # Classification report
                st.subheader("📋 Detailed Classification Report")
                report = classification_report(
                    data['test_labels'], 
                    results['test_predictions'],
                    target_names=['Negative', 'Positive']
                )
                st.text(report)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run ZS-SC Pipeline' to start")
        
        st.markdown("""
        ### About This Implementation
        
        This is a clean implementation of the ZS-SC (Zero-Shot Sentiment Classification) method from:
        
        **"Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts"**  
        *Mohna Chakraborty, Adithya Kulkarni, and Qi Li*  
        ACL 2023
        
        #### Pipeline Steps:
        1. **Generate Templates**: Uses positioning, subordination, and paraphrasing techniques
        2. **Score Templates**: Ranks templates based on sensitivity to keyword changes
        3. **Evaluate**: Tests top-ranked templates on the dataset
        
        #### Key Features:
        - ✅ Zero-shot learning (no training required)
        - ✅ Automatic prompt augmentation
        - ✅ Novel ranking metric
        - ✅ Multiple datasets (SST-2, MR, CR)
        - ✅ BERT-base and BERT-large support
        - ✅ **Subset sampling for faster processing**
        
        #### Performance Tips:
        - Use smaller subset sizes for quick testing (e.g., 100 train, 200 test)
        - Limit synonym samples to 20-30 for faster scoring
        - BERT-base is faster than BERT-large
        - GPU acceleration recommended for larger datasets
        """)


if __name__ == "__main__":
    main()
