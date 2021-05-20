# [Getting started with Google BERT](https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5)

### Build and train state-of-the-art natural language processing models using BERT 
## About the book
<a target="_blank" href="https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5">
  <img src="./images/book_cover.jpg" alt="Book Cover" width="300" align="left"/>
 
</a>BERT (bidirectional encoder representations from transformer) has revolutionized the world of natural language processing (NLP) with promising results. This book is an introductory guide that will help you get to grips with Google's BERT architecture. With a detailed explanation of the transformer architecture, this book will help you understand how the transformer's encoder and decoder work.

You'll explore the BERT architecture by learning how the BERT model is pre-trained and how to use pre-trained BERT for downstream tasks by fine-tuning it for NLP tasks such as sentiment analysis and text summarization with the Hugging Face transformers library. As you advance, you'll learn about different variants of BERT such as ALBERT, RoBERTa, and ELECTRA, and look at SpanBERT, which is used for NLP tasks like question answering. You'll also cover simpler and faster BERT variants based on knowledge distillation such as DistilBERT and TinyBERT.

The book takes you through MBERT, XLM, and XLM-R in detail and then introduces you to sentence-BERT, which is used for obtaining sentence representation. Finally, you'll discover domain-specific BERT models such as BioBERT and ClinicalBERT, and discover an interesting variant called VideoBERT.

## Get the book 
<div>
<a target="_blank" href="https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5">
  <img src="./images/Oreilly_safari_logo.png" alt="Oreilly Safari" hieght=150, width=150>
</a>
  
<a target="_blank" href="https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5">
  <img src="./images/amazon_logo.jpg" alt="Amazon" >
</a>

<a target="_blank" href="https://www.packtpub.com/product/getting-started-with-google-bert/9781838821593">
  <img src="./images/packt_logo.jpeg" alt="Packt" hieght=150, width=150 >
</a>

<a target="_blank" href="https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5">
  <img src="./images/googlebooks_logo.png" alt="Google Books" 
</a>

<a target="_blank" href="https://www.amazon.com/gp/product/B08LLDF377/ref=dbs_a_def_rwt_bibl_vppi_i5">
  <img src="./images/googleplay_logo.png" alt="Google Play" >
</a>
<br>
</div>
<br>


## Clone the repo and run in Google Colab  <img src="https://colab.research.google.com/img/colab_favicon_256px.png" hieght=50, width=50 >



### [1. A Primer on Transformer](1.%20A%20primer%20on%20transformer)

* [1.1. Introduction to transformer](1.%20A%20primer%20on%20transformer/1.01.%20Introduction%20to%20transformer.ipynb)
* [1.2. Understanding encoder of transformer](1.%20A%20primer%20on%20transformer/1.02.%20Understanding%20Encoder%20of%20transformer.ipynb)
* [1.3. Self-attention mechanism](1.%20A%20primer%20on%20transformer/1.03.%20Self-attention%20mechanism%20.ipynb)
* [1.4. Understanding self-attention mechansim](1.%20A%20primer%20on%20transformer/1.04.%20%20Understanding%20Self-attention%20mechanism.ipynb)
* 1.5. Multi-head attention mechanism
* 1.6. Learning position with positional encoding
* 1.7. Feedforward network
* 1.8. Add and norm component
* 1.9. Putting all encoder components together
* 1.10. Understanding decoder of transformer
* 1.11. Masked Multi-head attention
* 1.12. Feedforward Network
* 1.13. Add and norm component
* 1.14. Linear and softmax layer
* 1.15. Putting all decoder components together

### [2. Understanding the BERT model](2.%20Understanding%20BERT%20model)

* 2.1. Basic idea of BERT
* 2.2. Working of BERT
* 2.3. Configuration of BERT
* 2.4. Pre-training the BERT
* 2.5. Pre-training strategies
* 2.6. Pre-training procedure
* 2.7. Subword tokenization algorithms
* 2.8. Byte pair encoding
* 2.9. Byte-level byte pair encoding
* 2.10. WordPiece


### [3. Getting hands-on with BERT](3.%20Getting%20hands-on%20with%20BERT)

* 3.1. Pre-trained BERT model
* [3.2. Extracting embeddings from pre-trained BERT](3.%20Getting%20hands-on%20with%20BERT/3.02.%20Extracting%20embeddings%20from%20pre-trained%20BERT%20.ipynb)
* [3.3. Generating BERT embedding](3.%20Getting%20hands-on%20with%20BERT/3.03.%20Generating%20BERT%20embedding%20.ipynb)
* [3.4. Extracting embeddings from all encoder layers of BERT](3.%20Getting%20hands-on%20with%20BERT/3.04.%20Extracting%20embeddings%20from%20all%20encoder%20layers%20of%20BERT.ipynb)
* 3.5. Finetuning BERT for downstream tasks
* [3.6. Text classification](3.%20Getting%20hands-on%20with%20BERT/3.06.%20Text%20classification.ipynb)
* 3.7. Natural language inference
* 3.8. Question answering
* 3.9. Q&A with fine-tuned BERT
* 3.10. Named-entity recognition

### [4. BERT variants I - ALBERT, RoBERTa, ELECTRA, SpanBERT](4.%20BERT%20variants%20I%20-%20ALBERT%2C%20RoBERTa%2C%20ELECTRA%2C%20SpanBERT)

* 4.1. A Lite version of BERT
* 4.2. Training the ALBERT
* 4.3. Extracting embedding with ALBERT
* 4.4. Robustly Optimized BERT Pretraining Approach
* [4.5. Exploring the RoBERTa tokenizer](4.%20BERT%20variants%20I%20-%20ALBERT%2C%20RoBERTa%2C%20ELECTRA%2C%20SpanBERT/4.05.%20Exploring%20the%20RoBERTa%20tokenizer%20.ipynb)
* 4.6. Understanding ELECTRA
* 4.7. Generator and discriminator of ELECTRA
* 4.8. Training the ELECTRA model
* 4.9. Predicting span with SpanBERT 
* 4.10. Architecture of SpanBERT 
* 4.11. Exploring SpanBERT
* [4.12. Performing question-answering with pre-trained
SpanBERT](4.%20BERT%20variants%20I%20-%20ALBERT%2C%20RoBERTa%2C%20ELECTRA%2C%20SpanBERT/4.12.%20Performing%20question-answering%20with%20pre-trained%0ASpanBERT%20.ipynb)

### [5. BERT variants II - Based on knowledge distillation](5.%20BERT%20variants%20II%20-%20Based%20on%20knowledge%20distillation)

* 5.1. Knowledge distillation
* 5.2. DistilBERT - distilled version of BERT
* 5.3. Training the DistilBERT
* 5.4. TinyBERT
* 5.5. Teacher-student architecture
* 5.6. Training the TinyBERT
* 5.7. Transferring knowledge from BERT to neural network
* 5.8. Teacher-student architecture
* 5.9. Training the student network
* 5.10. Data augmentation method

### [6. Exploring BERTSUM for text summarization](6.%20Exploring%20BERTSUM%20for%20text%20summarization)

* 6.1. Text summarization
* 6.2. Fine-tuning BERT for text summarization
* 6.3. Extractive summarization using BERT
* 6.4. Abstractive summarization using BERT
* 6.5. Understanding ROUGE evaluation metric
* 6.6. Performance of BERTSUM model
* [6.7. Training the BERTSUM model](6.%20Exploring%20BERTSUM%20for%20text%20summarization/6.07.%20Training%20the%20BERTSUM%20model%20.ipynb)


### [7. Applying BERT for other languages](7.%20Applying%20BERT%20for%20other%20languages)

* 7.1. Understanding multilingual BERT
* 7.2. How multilingual is the multilingual BERT?
* 7.3. Cross-lingual language model
* 7.4. Understanding XLM-R
* 7.5. Language-specific BERT
* 7.6. FlauBERT for French
* [7.7. Getting representation of French sentence with FlauBERT](7.%20Applying%20BERT%20for%20other%20languages/7.07.%20Getting%20representation%20of%20French%20sentence%20with%20FlauBERT%20.ipynb)
* 7.8. BETO for Spanish
* [7.9. Predicting masked word using BETO](7.%20Applying%20BERT%20for%20other%20languages/7.09.Predicting%20masked%20word%20using%20BETO%20.ipynb)
* 7.10. BERTJe for Dutch
* [7.11. Next sentence prediction with BERTJe](7.%20Applying%20BERT%20for%20other%20languages/7.11.%20Next%20sentence%20prediction%20with%20BERTje.ipynb)
* 7.12. German, Chinese, and Japanese BERT 
* 7.13. understanding FinBERT and UmBERTo
* 7.14. BERTimbau for Portuguese
* 7.15. RuBERT for Russian 

### [8. Exploring Sentence and Domain Specific BERT](8.%20Exploring%20Sentence%20and%20Domain%20Specific%20BERT)

* 8.1. Learning sentence representation with Sentence-BERT
* 8.2. Understanding Sentence-BERT
* [8.3. Exploring sentence-transformers library](8.%20Exploring%20Sentence%20and%20Domain%20Specific%20BERT/8.03.%20Exploring%20sentence-transformers%20library%20.ipynb)
* 8.4. Computing sentence representation using Sentence-BERT
* [8.5. Computing sentence similarity](8.%20Exploring%20Sentence%20and%20Domain%20Specific%20BERT/8.05.%20Computing%20sentence%20similarity%20.ipynb)
* 8.6. Loading Custom models 
* [8.7. Finding a similar sentence with Sentence-BERT](8.%20Exploring%20Sentence%20and%20Domain%20Specific%20BERT/8.07.%20Finding%20a%20similar%20sentence%20with%20Sentence-BERT%20.ipynb)
* 8.8. Learning multilingual embeddings through knowledge distillation
* 8.9. Exploring Domain-specific BERT 
* 8.10. Clinical BERT 
* 8.11. BioBERT 

### [9. Understanding VideoBERT, BART, and more](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more)

* 9.1. Learning language and video representation with VideoBERT
* 9.2. Pre-training objective
* 9.3. Understanding BART
* 9.4. Noising techniques
* [9.5. Performing text summarization with BART](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.05.%20Performing%20text%20summarization%20with%20BART%20.ipynb)
* 9.6. Ktrain
* [9.7. Sentiment analysis using Ktrain](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.07.%20Sentiment%20analysis%20using%20Ktrain.ipynb)
* [9.8. Building a document answering model](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.08.%20Building%20a%20document%20answering%20model%20.ipynb)
* [9.9. Document summarization](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.09.%20Document%20summarization.ipynb)
* [9.10. Computing sentence representation using BERT as service](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.10.%20Computing%20sentence%20representation%20using%20BERT%20as%20service.ipynb)
* [9.11. Computing contextual word representation](9.%20Understanding%20VideoBERT%2C%20BART%2C%20and%20more/9.11.%20Computing%20contextual%20word%20representation%20.ipynb)

