from concurrent.futures import ThreadPoolExecutor
import itertools
import os
import json
import requests
from bs4 import BeautifulSoup
from termcolor import colored
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
import time

import dspy
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from urllib.parse import urlparse
from duckduckgo_search import DDGS

from utils import chunk_text, print_header, process_JSON_response

@dataclass
class Citation:
    snippet: str
    source_url: str
    source_title: str
    relevance_score: float

@dataclass
class Answer:
    text: str
    citations: List[Citation]

@dataclass
class ClaimComponent: 
    question_text: str
    search_queries: List[str]
    component_type: str = None
    answer: Optional[Answer] = None

@dataclass
class Claim:
    text: str
    questions: List[ClaimComponent] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

@dataclass
class RetrievedDocument:
    content: str
    metadata: Dict[str, str]
    score: float

@dataclass
class SearchResult:
    title: str
    url: str
    source: str
    excerpt: str = None # relevant excerpt provided by search provider
    timestamp: Optional[str] = None

class SearchProvider:
    def __init__(self, provider: str = "serper", api_key: str = None):
        self.session = requests.Session()
        self.provider = provider
        self.api_key = api_key
        
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        # Choose search provider
        if self.provider == "serper":
            results = self._serper_search(query, num_results)
        elif self.provider == "duckduckgo":
            results = self._duckduckgo_search(query, num_results)
        else:
            raise ValueError(f"Unsupported search provider: {self.provider}")
        
        # if VER`BOSE: print(f"Found {len(results)} results")

        filtered_results = self._filter_and_rank_results(results)

        return filtered_results

    def _serper_search(self, query: str, num_results: int) -> List[SearchResult]:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": num_results}

        try:
            response = self.session.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status() # Raise an exception for bad status codes
            results = response.json()["organic"]
            
            return [
                SearchResult(
                    title=result["title"],
                    url=result["link"],
                    source=urlparse(result["link"]).netloc.lower(),
                    excerpt=result.get("snippet", ""),
                    timestamp=result.get("date")
                )
                for result in results
            ]
        except requests.exceptions.RequestException as e:
            print(f"Search error: {str(e)}")
            return []
        
    def _duckduckgo_search(self, query: str, num_results: int) -> List[SearchResult]:
        num_attempts = 3
        wait_time = 2  # seconds between retries
        
        for attempt in range(num_attempts):
            try:
                results = DDGS().text(query, max_results=num_results)
                return [
                    SearchResult(
                        title=result["title"], 
                        url=result["href"], 
                        excerpt=result["body"],
                        source=urlparse(result["href"]).netloc.lower(),
                    )
                    for result in results
                ]
            except Exception as e:
                if "rate" in str(e).lower() and attempt < num_attempts - 1:
                    if VERBOSE:
                        print(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"DuckDuckGo search error: {str(e)}")
                    return []
        
    def _filter_and_rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter and rank search results based on various criteria."""
        # Remove duplicates
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Filter out low-quality results
        filtered_results = [
            result for result in unique_results
            if self._is_reliable_source(result.url) and
            len(result.excerpt) > 50  # Minimum snippet length
        ]
        
        # Sort by relevance (TODO: implement more sophisticated ranking)
        filtered_results.sort(
            key=lambda x: (
                self._is_reliable_source(x.url), # Prioritize reliable sources
                bool(x.timestamp),  # Prioritize results with timestamps
                len(x.excerpt),  # Longer snippets might be more informative
            ),
            reverse=True
        )
        
        return unique_results

    def _is_reliable_source(self, url: str) -> bool:
        """Check if the source is reliable."""
        reliable_domains = {
            'wikipedia.org',
            'reuters.com',
            'bloomberg.com',
            'nytimes.com',
            'wsj.com',
            'science.org',
            'nature.com',
            'economist.com',
            'forbes.com',
            'apnews.com',
        }
        
        domain = urlparse(url).netloc.lower()
        if '.gov' in domain: return True
        return any(rd in domain for rd in reliable_domains)
    
class VectorStore:
    def __init__(
        self, 
        model_name: str,
        max_chunk_size: int = 1000,
        max_chunk_overlap: int = 100,
        use_bm25: bool = False,
        bm25_weight: float = 0.5
    ):
        self.encoder = SentenceTransformer(model_name)
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.max_chunk_size = max_chunk_size
        self.max_chunk_overlap = max_chunk_overlap

        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        if use_bm25: self.bm25 = None
            
    def add_documents(self, documents: List[str], metadata: List[Dict[str, str]]):
        if len(documents) == 0: return

        # Process documents in parallel
        with ThreadPoolExecutor() as executor:
            chunked_docs = list(
                executor.map(
                    chunk_text, 
                    documents, 
                    itertools.repeat(self.max_chunk_size), 
                    itertools.repeat(self.max_chunk_overlap)
                )
            )

        # Flatten chunks and duplicate metadata
        all_chunks = []
        all_metadata = []
        for chunks, meta in zip(chunked_docs, metadata):
            all_chunks.extend(chunks)
            all_metadata.extend([meta] * len(chunks))
        
        # Encode chunks
        embeddings = self.encoder.encode(
            all_chunks,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Initialize or update FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.cpu().numpy())
        self.documents.extend(all_chunks)
        self.metadata.extend(all_metadata)
        
        # Initialize BM25 index
        if self.use_bm25:
            tokenized_docs = [doc.split() for doc in all_chunks]
            self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, k: int = 10) -> List[RetrievedDocument]:
        # Get FAISS scores
        query_embedding = self.encoder.encode([query])[0].reshape(1, -1) # Reshape to 1D array
        faiss_scores, faiss_indices = self.index.search(query_embedding, k)
        
        # Get relevant chunks based on FAISS scores (and combine w/ BM25 scores if enabled)
        results = []
        for i, idx in enumerate(faiss_indices[0]):
            score = float(faiss_scores[0][i])
            
            # Combine FAISS and BM25 scores using weighted average
            # if self.use_bm25:
            #     bm25_score = self.bm25.get_scores([query])[idx]
            #     score = (
            #         self.bm25_weight * bm25_score + 
            #         (1 - self.bm25_weight) * score
            #     )

            # Save retrieved document w/ weighted score
            if score > 0: # TODO: make this threshold configurable
                results.append(
                    RetrievedDocument(
                        content=self.documents[idx],
                        metadata=self.metadata[idx],
                        score=score
                    )
                )
        
        # Sort results by score in descending order
        return sorted(results, key=lambda x: x.score, reverse=True)

class ClaimExtractorSignature(dspy.Signature):
    """Extract specific, testable factual claims from the given text."""
    # """
    #     Task: Extract specific, testable factual claims from the given text.
    #     Requirements:
    #     1. Each claim should be context-independent
    #     2. Focus on single, verifiable ideas
    #     3. Maintain clarity and coherence
    #     4. If a statement is atomic, keep it as is
    # """
    text = dspy.InputField(desc="The text to extract claims from")
    claims = dspy.OutputField(desc="""JSON object containing:
    {
        "claims": [
            {
                "text": string,
                "reasoning": string
            }
        ]
    }""")

class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimExtractorSignature)

    def forward(self, text: str) -> List[Claim]:
        # Extract claims with LLM (retry if failed)
        try:
            result = self.extract(text=text)
            claims = process_JSON_response(result["claims"])["claims"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(text)
        
        # If verbose, print extracted claims
        if VERBOSE or INTERACTIVE:
            print_header(f"Extracted Claims ({len(claims)}): ", level=1)
            for i, claim in enumerate(claims, 1):
                print_header(f"{i}. {colored(claim['text'], 'white')}", level=2, decorator='')
                # print_header(f"  Reasoning: {claim['reasoning']}", level=3)
        
        # If interactive, allow user to provide feedback to regenerate claims
        if INTERACTIVE:
            while True:
                try:
                    feedback = input("Are these claims correctly extracted? (yes/no): ").lower()
                    if feedback == "yes":
                        break
                    elif feedback == "no":
                        feedback = input("Please provide feedback on what's wrong: ")
                        # Regenerate claims with feedback (TODO: implement history)
                        text = f"{text} {feedback}"
                        return self.forward(text=text)
                        # claims = json.loads(result)["claims"]
                        # print("\nRegenerated Claims:")
                        # for i, claim in enumerate(claims, 1):
                        #     print(f"{i}. {claim['text']}")
                        #     print(f"   Reasoning: {claim['reasoning']}\n")
                except KeyboardInterrupt:
                    print("\nProcess interrupted. Exiting...")
                    exit(0)
        
        return [Claim(text=claim["text"]) for claim in claims]

class ClaimDecomposerSignature(dspy.Signature):
    """Break down a claim to generate independent questions and search queries to answer it. Be as specific and concise as possible, try to minimize the number of questions and search queries while still being comprehensive to verify the claim."""
    claim = dspy.InputField(desc="The claim to decompose into components (questions + search queries)")
    questions = dspy.OutputField(desc="""JSON object containing:
    {
        "questions": [
            {
                "question_text": string, # question text (e.g. "What was the GDP growth rate during the Trump administration?")
                "search_queries": [string], # independent search queries used to answer the question, try to be as specific as possible and avoid redundancy, no more than 2 queries
            }
        ]
    }""")
                # "component_type": string # type of question (e.g. "metric", "time_period", "comparison", "causal_relation")

class ClaimDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(ClaimDecomposerSignature)

    def forward(self, claim: Claim) -> List[ClaimComponent]:
        # Decompose claim into components (questions + search queries) with LLM (retry if failed)
        try:
            result = self.decompose(claim=claim.text)
            data = process_JSON_response(result["questions"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(claim=claim)
        
        # If verbose, print generated components
        if VERBOSE or INTERACTIVE:
            print_header(f"Decomposed Components (Questions + Search Queries) ({len(data['questions'])}):", level=3)
            for i, component in enumerate(data['questions'], 1):
                print_header(f"{i}. Question: {colored(component['question_text'], 'yellow')}", level=4)
                print_header(f"   Search Queries: {colored(component['search_queries'], 'yellow')}", level=4)

        # If interactive, allow user to provide feedback to regenerate components
        if INTERACTIVE:
            while True:
                feedback = input("Are these components correct? (yes/no): ").lower()
                if feedback == "yes":
                    break
                elif feedback == "no":
                    feedback = input("Please provide feedback on what's wrong: ")
                    # Regenerate components with feedback (TODO: implement history)
                    result = self.forward(claim=f"{claim.text} {feedback}")    
                    # data = process_JSON_response(result["components"])
                    # print("\nRegenerated Components:")
                    # print(json.dumps(data, indent=2))

        return [ClaimComponent(**component) for component in data['questions']]
    
## ANSWER SYNTHESIZER ##
class AnswerSynthesizerSignature(dspy.Signature):
    """Synthesize an answer based on retrieved documents with inline citations."""
    question = dspy.InputField(desc="The question to answer")
    search_queries = dspy.InputField(desc="The search queries used")
    documents = dspy.InputField(desc="Retrieved documents relevant to the question")
    answer = dspy.OutputField(desc="""JSON object containing: 
    {
        "text": string, # answer with inline citations (e.g., "The wage gap was shrinking [1]")
        "citations": [{ # list of citations
            "snippet": string,  # exact quote from source
            "source_url": string,
            "source_title": string,
            "relevance_score": float
        }]
    }""")

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(AnswerSynthesizerSignature)

    def forward(self, component: ClaimComponent, documents: List[RetrievedDocument]) -> Answer:
        if VERBOSE:
            print_header("Question: " + colored(component.question_text, 'yellow'), level=4)
            print_header("Search Queries: " + colored(component.search_queries, 'yellow'), level=4)

        # Synthesize answer with LLM (retry if failed)
        try: 
            result = self.synthesize(
                question=component.question_text,
                search_queries=component.search_queries,
                documents=[{
                    "title": doc.metadata["title"],
                    "url": doc.metadata["url"],
                    "content": doc.content
                } for doc in documents]
            )
            data = process_JSON_response(result["answer"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(component, documents)
        
        answer = Answer(
            text=data["text"],
            citations=[Citation(**citation) for citation in data["citations"]]
        )

        if VERBOSE:
            print_header(f"Answer: {colored(answer.text, 'green')}", level=4)
            print_header("Citations: ", level=4)
            for i, citation in enumerate(answer.citations, 1):
                print_header(f"[{i}] " + colored(citation.snippet, 'yellow'), level=5)
                print_header(f"Source: {citation.source_title} ({citation.source_url})", level=5)

        return answer
    
## CLAIM EVALUATOR ##
class ClaimEvaluatorSignature(dspy.Signature):
    """Evaluate a claim based on questions and answers."""
    claim = dspy.InputField(desc="The claim to evaluate")
    qa_pairs = dspy.InputField(desc="Question-answer pairs with citations")
    evaluation = dspy.OutputField(desc="""JSON object containing:
    {
        "verdict": string,  # One of: TRUE, MOSTLY TRUE, HALF TRUE, MOSTLY FALSE, FALSE, UNVERIFIABLE
        "confidence": float,  # Between 0 and 1
        "reasoning": string,
        "evidence_analysis": [
            {
                "question": string,
                "answer": string,
                "contribution": string
            }
        ]
    }""")

class ClaimEvaluator(dspy.Module):
    VERDICTS = [
        "TRUE", "MOSTLY TRUE", "HALF TRUE", 
        "MOSTLY FALSE", "FALSE", "UNVERIFIABLE"
    ]
    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(ClaimEvaluatorSignature)

    def forward(self, claim: Claim) -> Tuple[str, float, str]:
        qa_pairs = [
            {
                "question": c.question_text,
                "answer": c.answer.text,
                # "citations": [vars(c) for c in a.citations]
            }
            for c in claim.components
        ]
        
        result = self.evaluate(
            claim=claim.text,
            qa_pairs=json.dumps(qa_pairs, indent=2)
        )
        data = process_JSON_response(result["evaluation"])
        verdict = data["verdict"]
        confidence = data["confidence"]
        reasoning = data["reasoning"]
        
        if VERBOSE or INTERACTIVE:
            print_header("Claim Evaluation", level=2, decorator='=')
            print_header(f"Claim: {colored(claim.text, 'white')}", level=3)
            print_header(f"Verdict: {colored(verdict, 'green')}", level=3)
            print_header(f"Confidence: {colored(str(confidence), 'yellow')}", level=3)
            print_header(f"Reasoning: {colored(reasoning, 'cyan')}", level=3)
            print_header("Evidence Analysis", level=3)
            for i, evidence in enumerate(data["evidence_analysis"], 1):
                print_header(colored(f"Question: {evidence['question']}", 'yellow'), level=4)
                print_header(colored(f"Answer: {evidence['answer']}", 'green'), level=4)
                print_header(colored(f"Contribution: {evidence.get('contribution', '')}", 'cyan'), level=4)
            
        if INTERACTIVE:
            while True:
                feedback = input("Is this evaluation correct? (yes/no): ").lower()
                if feedback == "yes":
                    break
                elif feedback == "no":
                    feedback = input("Please provide feedback on what's wrong: ")
                    return self.forward(claim=f"{claim.text} {feedback}")
        
        return verdict, confidence, reasoning

class OverallVerdictSignature(dspy.Signature):
    """Calculate overall verdict based on atomic claims."""
    claims = dspy.InputField(desc="List of evaluated atomic claims")
    overall_verdict = dspy.OutputField(desc="""JSON object containing:
    {
        "verdict": string,  # One of: TRUE, MOSTLY TRUE, HALF TRUE, MOSTLY FALSE, FALSE, UNVERIFIABLE
        "confidence": float,
        "reasoning": string
    }""")

class FactCheckPipeline:
    def __init__(
        self,
        search_provider: SearchProvider,
        model_name: str,
        embedding_model: str,
        retriever_k: int = 10
    ):
        self.search_provider = search_provider
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # Initialize components
        self.claim_extractor = ClaimExtractor()
        self.claim_decomposer = ClaimDecomposer()
        self.retriever = VectorStore(
            model_name=embedding_model,
            use_bm25=USE_BM25,
            bm25_weight=BM25_WEIGHT
        )
        self.answer_synthesizer = AnswerSynthesizer()
        self.claim_evaluator = ClaimEvaluator()
        
        # Chat history for interactive mode, TODO: implement history
        self.chat_history = []
        
    def _scrape_webpage(self, url: str) -> Tuple[str, Dict[str, str]]:
        try:
            response = requests.get(url, timeout=SCRAPE_TIMEOUT)  # Set a timeout of 5 seconds
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (customize based on website structure)
            content = ' '.join([p.text for p in soup.find_all('p')])
            
            metadata = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "source": urlparse(url).netloc
            }
            
            return content, metadata
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return "", {}

    def _index_into_retriever(self, query: str, search_results: List[str]) -> List[RetrievedDocument]:  # TODO: combine this   
        # Process webpages and index content
        documents = []
        metadata = []
        for result in tqdm(search_results, desc="Processing sources"):
            # First add source relevant excerpt to documents if available
            if result.excerpt:
                documents.append(result.excerpt)
                metadata.append({
                    "title": result.title,
                    "url": result.url,
                    "source": result.source
                })

            # Otherwise, scrape the webpage
            # content, meta = self._process_webpage(result.url)
            # if content: 
            #     # Split content into chunks, preserving full sentences
            #     chunks = chunk_text(content)
            #     documents.extend(chunks)
            #     metadata.extend([meta] * len(chunks))
        
        self.retriever.add_documents(documents, metadata)
        return self.retriever.retrieve(query, k=self.retriever_k)

    def fact_check(self, statement: str) -> List[Claim]:
        if VERBOSE:
            print_header("Starting Fact Check Pipeline", level=0, decorator='=')
            print_header(f"Original Statement: {colored(statement, 'white')}", level=0)

        # Step 1: Extract atomic claims
        if VERBOSE: print_header("Atomic Claim Extraction", level=1, decorator='=')
        claims = self.claim_extractor(statement)
        for claim_i, claim in enumerate(claims, 1):
            # Step 2: Decompose claim into components (questions and search queries)
            if VERBOSE: print_header(f"Claim Decomposition [{claim_i}/{len(claims)}]", level=2, decorator='=')
            components = self.claim_decomposer(claim) # List of ClaimComponent objects
            for component_i, component in enumerate(components, 1):
                if VERBOSE:
                    print_header(f"Question Answering for Component [{component_i}/{len(components)}]", level=3, decorator='=')
                    print_header(f"Question: {colored(component.question_text, 'yellow')}", level=4)
                    print_header(f"Search Queries: {colored(component.search_queries, 'yellow')}", level=4)

                # Step 3: Search and retrieve
                relevant_docs = []
                for query_i, query in enumerate(component.search_queries, 1): 
                    if VERBOSE: print_header(f"Web Search for Query [{query_i}/{len(component.search_queries)}]", level=4, decorator='=')
                    print_header(f"Query: {colored(query, 'yellow')}", level=4)
                    # Perform web search
                    search_results = self.search_provider.search(query, NUM_SEARCH_RESULTS)

                    if VERBOSE:
                        print_header(f"Retrieved {len(search_results)} Sources:", level=4)
                        for i, result in enumerate(search_results, 1):
                            print_header(f"{i}. {result.title}", level=5)
                            print_header(f"URL: {result.url}", level=5)
                            print_header(f"Excerpt: {result.excerpt}", level=5)

                    if INTERACTIVE:
                        while True:
                            feedback = input("Are these sources relevant? (yes/no): ").lower()
                            if feedback == "yes":
                                break
                            elif feedback == "no":
                                feedback = input("Please provide feedback on what's wrong: ")
                                search_results = self.search_provider.search(
                                    f"{query} {feedback}"
                                )
                                continue
                    
                    # Get relevant documents
                    docs = self._index_into_retriever(query, search_results)
                    relevant_docs.extend(docs)

                # Step 4: Synthesize answer
                if VERBOSE: print_header(f"Synthesizing Answer [{component_i}/{len(components)}]", level=3, decorator='=')
                answer = self.answer_synthesizer(
                    component=component,
                    documents=relevant_docs
                )
                component.answer = answer # Set answer to the question

            # Set claim components (questions and answers)
            claim.components = components

            # Step 5: Evaluate claim
            verdict, confidence, reasoning = self.claim_evaluator(claim=claim)
            # Set claim attributes
            claim.verdict = verdict
            claim.confidence = confidence
            claim.reasoning = reasoning

        if VERBOSE:
            print_header("Final Results", level=0, decorator='=')
            for i, claim in enumerate(claims, 1):
                print_header(f"Claim {i}", level=1)
                print_header("Text: " + colored(claim.text, 'white'), level=2)
                print_header("Verdict: " + colored(claim.verdict, 'green'), level=2)
                print_header("Confidence: " + colored(str(claim.confidence), 'yellow'), level=2)
                print_header("Reasoning: " + colored(claim.reasoning, 'cyan'), level=2)
                
                for j, component in enumerate(claim.components, 1):
                    print_header(f"Component {j}", level=2)
                    print_header("Question: " + colored(component.question_text, 'yellow'), level=3)
                    print_header("Answer: " + colored(component.answer.text, 'green'), level=3)
                    print_header("Citations: ", level=3)
                    for k, citation in enumerate(component.answer.citations, 1):
                        print_header(f"[{k}] " + colored(citation.snippet, 'yellow'), level=4)
                        print_header(f"Source: {citation.source_title} ({citation.source_url})", level=4)

        return claims

# Example usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv('../.env')

    # Initialize search provider
    NUM_SEARCH_RESULTS = 10 # Number of search results to retrieve
    SCRAPE_TIMEOUT = 5 # Timeout for scraping a webpage (in seconds)
    search_provider = SearchProvider(provider="duckduckgo")

    # Initialize DSPy
    lm = dspy.LM('gemini/gemini-1.5-flash', api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
    # lm = dspy.LM('ollama/mistral')
    dspy.settings.configure(lm=lm)

    # Initialize pipeline
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    VERBOSE = True # Print intermediate results
    INTERACTIVE = False # Allow the user to provide feedback
    USE_BM25 = True # Use BM25 for retrieval (in addition to cosine similarity)
    BM25_WEIGHT = 0.5 # Weight for BM25 in the hybrid retrieval

    pipeline = FactCheckPipeline(
        search_provider=search_provider,
        model_name=lm,
        embedding_model=embedding_model,
        retriever_k=2
    )

    # Example statement to fact-check
    # statement = """And then there's the reality of the Trump economy, 
    # where wages adjusted for inflation were rising. The wage gap between 
    # rich and poor was shrinking. The savings rate for black Americans was 
    # the highest in the history of our country."""

    statement = """The US economy is in a recession now in 2024."""

    # Run fact-checking pipeline
    result = pipeline.fact_check(statement)

    # Print final result
    print("\nFinal Fact-Check Result:")
    print(result)
    # print(json.dumps(result, indent=2))