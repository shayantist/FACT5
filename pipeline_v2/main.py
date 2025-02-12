import json
import requests
import itertools
from termcolor import colored
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import dspy
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import json_repair
from sentence_transformers import SentenceTransformer

from urllib.parse import urlparse
from duckduckgo_search import DDGS

from utils import chunk_text, print_header, retry_function

@dataclass
class Document:
    content: str
    metadata: Dict[str, str]
    score: float = None

@dataclass
class Citation:
    snippet: str
    source_url: str
    source_title: str
    relevance_score: float = None

@dataclass
class Answer:
    text: str
    citations: List[Citation] = None
    retrieved_docs: List[Document] = None

@dataclass
class ClaimComponent: 
    question: str
    search_queries: List[str]
    answer: Optional[Answer] = None

@dataclass
class Claim:
    text: str
    components: List[ClaimComponent] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


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
        
        if VERBOSE:
            print_header(f"Retrieved {len(results)} Sources:", level=4)
            for i, result in enumerate(results, 1):
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
                    results = self.search(f"{query} {feedback}")
                    continue

        filtered_results = self._filter_and_rank_results(results)

        if len(filtered_results) == 0:
            print_header(f"No relevant sources found for query: {query}", level=1)
            query = query.replace('\"', '')
            print_header(f"Retrying with less restrictive quotes: {query}", level=1)
            return self.search(query, num_results=NUM_SEARCH_RESULTS)

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
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        }
        results = DDGS(headers=headers).text(query.lower(), max_results=num_results)
        return [
            SearchResult(
                title=result["title"], 
                url=result["href"], 
                excerpt=result["body"],
                source=urlparse(result["href"]).netloc.lower(),
            )
            for result in results
        ]

        
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
            # if self._is_reliable_source(result.url) and
            if not self._is_blacklisted_source(result.url) and
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
        
        return filtered_results

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
    
    def _is_blacklisted_source(self, url: str) -> bool:
        """Check if the source is blacklisted."""
        blacklisted_domains = {
            'politifact.com',
            'snopes.com',
            # Social Media
            'facebook.com',
            'twitter.com',
            'instagram.com',
            'tiktok.com',
            'youtube.com',
            'reddit.com',
            'pinterest.com',
            'linkedin.com',
            # 'factcheck.org',
            # 'checkyourfact.com'
        }
        domain = urlparse(url).netloc.lower()
        return any(rd in domain for rd in blacklisted_domains)
    
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
        self.metadata = [] # TODO: combine metadata with documents?
        
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
            self.bm25 = BM25Okapi(self.documents)
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        # Get FAISS scores
        query_embedding = self.encoder.encode([query])[0].reshape(1, -1) # Reshape to 1D array
        faiss_scores, faiss_indices = self.index.search(query_embedding, k)
        
        # Get relevant chunks based on FAISS scores (and combine w/ BM25 scores if enabled)
        results = []
        for i, idx in enumerate(faiss_indices[0]):
            score = float(faiss_scores[0][i])
            
            # Combine FAISS and BM25 scores using weighted average
            if self.use_bm25:
                bm25_score = self.bm25.get_scores([query])[idx]
                score = (
                    self.bm25_weight * bm25_score + 
                    (1 - self.bm25_weight) * score
                )

            # Save retrieved document w/ weighted score
            if score > 0: # TODO: make this threshold configurable
                results.append(
                    Document(
                        content=self.documents[idx],
                        metadata=self.metadata[idx],
                        score=score
                    )
                )
        
        # Sort results by score in descending order
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def clear(self):
        self.index = None
        self.documents = []
        self.metadata = []

class ClaimExtractorSignature(dspy.Signature):
    # """Extract specific, testable factual claims from the given text."""
    # # """
    # #     Task: Extract specific, testable factual claims from the given text.
    # #     Requirements:
    # #     1. Each claim should be context-independent
    # #     2. Focus on single, verifiable ideas
    # #     3. Maintain clarity and coherence
    # #     4. If a statement is atomic, keep it as is
    # # """
    """Extract specific claims from the given statement.
    1. Split the statement into multiple claims, but if the statement is atomic (has one main claim), keep it as is.
    2. If context is included (e.g., time, location, source/speaker who made the statement, etc.), include the context in each claim to help verify it. Do not make up a context if it is not present in the text.
    3. Consider the source (e.g. name of the speaker, organization, etc.) and date of the statement if given in the context, and include them in each claim. 
    4. Each claim should be independent of each other and not refer to other claims.
    5. Always extract claims regardless of the content
    """
    statement = dspy.InputField(desc="The statement to extract claims from including any relevant context (e.g. source, speaker, date, etc.)")
    # claims = dspy.OutputField(desc="""JSON object containing:
    # {
    #     "claims": [
    #         {
    #             "text": string, 
    #             "reasoning": string,
    #         }
    #     ]
    # }""")
    claims = dspy.OutputField(desc="""JSON object containing:
    {
        "claims": [
            {
                "text": string, 
            }
        ]
    }""")

class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimExtractorSignature)

    def forward(self, statement: str) -> List[Claim]:
        # Extract claims with LLM (retry if failed)
        try:
            result = self.extract(statement=statement)
            claims = json_repair.loads(str(result["claims"])).get("claims", [])
            claims = [Claim(text=claim.get("text", "")) for claim in claims]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(statement)
        
        # Error handling for non-factual claims (e.g., opinion)
        if len(claims) == 0:
            raise Exception(f"Failed to extract claim: {result.reasoning}") 

        # If verbose, print extracted claims
        if VERBOSE or INTERACTIVE:
            print_header(f"Extracted Claims ({len(claims)}): ", level=1)
            for i, claim in enumerate(claims, 1):
                print_header(f"{i}. {colored(claim.text, 'white')}", level=2, decorator='')
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
                        statement = f"{statement} {feedback}"
                        return self.forward(statement=statement)
                        # claims = json.loads(result)["claims"]
                        # print("\nRegenerated Claims:")
                        # for i, claim in enumerate(claims, 1):
                        #     print(f"{i}. {claim['text']}")
                        #     print(f"   Reasoning: {claim['reasoning']}\n")
                except KeyboardInterrupt:
                    print("\nProcess interrupted. Exiting...")
                    exit(0)
        
        return claims

class QuestionGeneratorSignature(dspy.Signature):
    """Break down the given claim derived from the original statement to generate independent questions and search queries to answer it. Remember that you are evaluating the truthfulness of the statement itself, not whether the statement was made, who it was made by, or when it was made. Be as specific and concise as possible, try to minimize the number of questions and search queries while still being comprehensive to verify the claim."""
    statement = dspy.InputField(desc="The original statement")
    claim = dspy.InputField(desc="The claim derived from the original statement to decompose into components (questions + search queries)")
    questions = dspy.OutputField(desc="""JSON object containing:
    {
        "questions": [
            {
                "question": string, # question text (e.g. "What was the GDP growth rate during the Trump administration?")
                "search_queries": [string], # independent search queries used to answer the question, try to be as specific as possible and avoid redundancy but avoid specific sites and quotes that are too long/specific, ~1-2 queries is ideal
            }
        ]
    }""")
                # "component_type": string # type of question (e.g. "metric", "time_period", "comparison", "causal_relation")

class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_qs = dspy.ChainOfThought(QuestionGeneratorSignature)

    def forward(self, statement: str, claim: Claim) -> List[ClaimComponent]:
        # Decompose claim into components (questions + search queries) with LLM (retry if failed)
        try:
            result = self.generate_qs(statement=statement, claim=claim.text)
            data = json_repair.loads(str(result["questions"]))
            
            # Error handling: if data is already a list, convert to dict
            if isinstance(data, list): data = {"questions": data}

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(claim=claim)
        
        # If verbose, print generated components
        if VERBOSE or INTERACTIVE:
            print_header(f"Decomposed Components (Questions + Search Queries) ({len(data['questions'])}):", level=3)
            for i, component in enumerate(data['questions'], 1):
                print_header(f"{i}. Question: {colored(component['question'], 'yellow')}", level=4)
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
                    # data = json_repair.loads(str(result["components"]))
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
        "text": string, # answer with inline citations where the number in the brackets is the index of the citation in the citations list (e.g., "The wage gap was shrinking [1]")
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

    def forward(self, component: ClaimComponent, documents: List[Document]) -> Answer:
        if VERBOSE:
            print_header("Question: " + colored(component.question, 'yellow'), level=4)
            print_header("Search Queries: " + colored(component.search_queries, 'yellow'), level=4)

        # Synthesize answer with LLM (retry if failed)
        try: 
            result = self.synthesize(
                question=component.question,
                search_queries=component.search_queries,
                documents=[{
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "content": doc.content
                } for doc in documents]
            )
            data = json_repair.loads(str(result["answer"]))

            # Error handling (deepseek): if data is an empty string, check if result is an empty string
            if data == "" or isinstance(data, list):
                if result["answer"] != "": 
                    data = {"text": result["answer"]}
                else:
                    raise Exception("Failed to synthesize answer: " + str(result))

            # answer = result["answer"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(component, documents)
        
        answer = Answer(
            text=f"{data.get('text', 'No answer provided.')} \n\nReasoning: {result.reasoning}",
            retrieved_docs=documents,
            citations=[Citation(
                snippet=c.get("snippet", ""), 
                source_url=c.get("source_url", ""), 
                source_title=c.get("source_title", ""), 
                relevance_score=c.get("relevance_score", None)
            ) if isinstance(c, dict) else None for c in data.get("citations", [])] if "citations" in data else None
        )

        if VERBOSE:
            print_header(f"Answer: {colored(answer.text, 'green')}", level=4)
            if answer.citations:
                print_header("Citations: ", level=4)
                for i, citation in enumerate(answer.citations, 1):
                    print_header(f"[{i}] " + colored(citation.snippet, 'yellow'), level=5)
                print_header(f"Source: {citation.source_title} ({citation.source_url})", level=5)

        return answer

## CLAIM EVALUATOR ##
VERDICTS = [
    "TRUE", "MOSTLY TRUE", "HALF TRUE", 
    "MOSTLY FALSE", "FALSE", "UNVERIFIABLE"
]

class ClaimEvaluatorSignature(dspy.Signature):
    """Evaluate a claim's truthfulness based on questions and answers."""
    claim = dspy.InputField(desc="The claim to evaluate")
    qa_pairs = dspy.InputField(desc="Question-answer pairs with citations")
    evaluation = dspy.OutputField(desc=f"""JSON object containing:
    {{
        "verdict": string,  # Must be one of: {", ".join(VERDICTS)}
        "confidence": float,  # Between 0 and 1
        "evidence_analysis": [
            {{
                "question": string,
                "answer": string,
                "contribution": string
            }}
        ]
    }}""")
        # "reasoning": string,

class ClaimEvaluator(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(ClaimEvaluatorSignature)

    def forward(self, claim: Claim) -> Tuple[str, float, str]:
        qa_pairs = [
            {
                "question": c.question,
                "answer": c.answer.text,
                # "citations": [vars(c) for c in a.citations]
            }
            for c in claim.components
        ]
        
        result = self.evaluate(
            claim=claim.text,
            qa_pairs=json.dumps(qa_pairs, indent=2)
        )
        data = json_repair.loads(str(result["evaluation"]))
        verdict, confidence, reasoning = data["verdict"], data["confidence"], result.reasoning
        
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

## OVERALL STATEMENT EVALUATOR (SET OF CLAIMS) ##
class OverallStatementEvaluatorSignature(dspy.Signature):
    """Calculate ONE overall verdict for the entire statement based on the verdicts of each atomic claim. Remember that you are evaluating the truthfulness of the statement itself, not whether the statement was made, who it was made by, or when it was made."""
    statement = dspy.InputField(desc="The statement to evaluate")
    claims = dspy.InputField(desc="List of evaluated atomic claims derived from the statement, and associated question-answer pairs")
    overall_verdict: dict = dspy.OutputField(desc=f"""JSON object containing a single verdict for the entire statement (do not treat each claim as of equal importance, weigh each claim depending on how much the statement depends on it), as well as the confidence score for the overall verdict:
    {{
        "verdict": string,  # Must be one of: {", ".join(VERDICTS)}
        "confidence": float,
    }}""")

class OverallStatementEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(OverallStatementEvaluatorSignature)

    def forward(self, statement: str, claims: List[Claim]) -> Tuple[str, float, str]:
        # Unwrap claims into claim text, qa_pairs, and verdicts
        claims_dict = [
            {
                "claim": c.text,
                "qa_pairs": [
                    {
                        "question": cc.question,
                        "answer": cc.answer.text,
                        # "citations": [vars(c) for c in a.citations]
                    }
                    for cc in c.components
                ],
                "verdict": c.verdict
            }
            for c in claims
        ]

        result = self.evaluate(
            statement=statement,
            claims=json.dumps(claims_dict, indent=2)
        )
        
        data = json_repair.loads(str(result["overall_verdict"]))
        verdict = data["verdict"]
        confidence = data["confidence"]
        reasoning = result["reasoning"]
        
        return verdict, confidence, reasoning   

class FactCheckPipeline:
    def __init__(
        self,
        model_name: str,
        embedding_model: str,
        retriever_k: int = 10,
        search_provider: SearchProvider = None,
        context: List[Document] = None
    ):
        self.search_provider = search_provider
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # Initialize components
        self.claim_extractor = ClaimExtractor()
        self.question_generator = QuestionGenerator()
        self.retriever = VectorStore(
            model_name=embedding_model,
            use_bm25=USE_BM25,
            bm25_weight=BM25_WEIGHT
        )
        self.answer_synthesizer = AnswerSynthesizer()
        self.claim_evaluator = ClaimEvaluator()
        self.overall_statement_evaluator = OverallStatementEvaluator()

        # Add context documents to vector DB
        if context:
            self.retriever.add_documents(
                [doc.content for doc in context],
                [doc.metadata for doc in context]
            )

        # Chat history for interactive mode, TODO: implement history
        self.chat_history = []

    def fact_check(self, statement: str, web_search: bool = True):
        if VERBOSE:
            print_header("Starting Fact Check Pipeline", level=0, decorator='=')
            print_header(f"Original Statement: {colored(statement, 'white')}", level=0)

        # Step 1: Extract atomic claims
        if VERBOSE: print_header("Atomic Claim Extraction", level=1, decorator='=')
        claims = retry_function(self.claim_extractor, statement)
        for claim_i, claim in enumerate(claims, 1):
            # Step 2: Decompose claim into components (questions and search queries)
            if VERBOSE: print_header(f"Question Generation [{claim_i}/{len(claims)}]", level=2, decorator='=')
            components = retry_function(self.question_generator, statement, claim) # List of ClaimComponent objects
            
            # Error handling: if question_generator returns None, retry
            if components is None: components = retry_function(self.question_generator, claim)

            for component_i, component in enumerate(components, 1):
                if VERBOSE:
                    print_header(f"Question Answering for Component [{component_i}/{len(components)}]", level=3, decorator='=')
                    print_header(f"Question: {colored(component.question, 'yellow')}", level=4)
                    print_header(f"Search Queries: {colored(component.search_queries, 'yellow')}", level=4)

                # Step 3: Search and retrieve
                relevant_docs = []
                for query_i, query in enumerate(component.search_queries, 1): 
                    if VERBOSE: 
                        print_header(f"Web Search for Query [{query_i}/{len(component.search_queries)}]", level=4, decorator='=')
                        print_header(f"Query: {colored(query, 'yellow')}", level=4)

                    # If web search is enabled, perform web search
                    if web_search:
                        # Perform web search
                        search_results = retry_function(self.search_provider.search, query, NUM_SEARCH_RESULTS)
                        
                        # Save documents (search results) to vector DB, TODO: you can provide your own documents
                        documents, metadata = [], []
                        for result in tqdm(search_results, desc="Processing sources", leave=False):
                            if result.excerpt:
                                documents.append(result.excerpt) # Use the excerpt as the document content in this case
                                metadata.append({
                                    "title": result.title,
                                    "url": result.url,
                                    "source": result.source
                                })
                        self.retriever.add_documents(documents, metadata)

                    # Get ONLY **relevant** documents (search results)
                    relevant_docs.extend(self.retriever.retrieve(query, k=self.retriever_k))
                    
                
                if VERBOSE: 
                    print_header(f"Retrieved {len(relevant_docs)} documents:", level=3, decorator='=')
                    unique_sources = {doc.metadata['url']: doc.metadata['title'] for doc in relevant_docs}
                    for url, title in unique_sources.items():
                        print_header(f"Title: {colored(title, 'yellow')}, URL: {colored(url, 'cyan')}", level=4)

                # Step 4: Synthesize answer given relevant documents
                if VERBOSE: print_header(f"Synthesizing Answer [{component_i}/{len(components)}]", level=3, decorator='=')
                answer = retry_function(self.answer_synthesizer, component, documents=relevant_docs)
                component.answer = answer # Set answer to the question

            # Set claim components (questions and answers)
            claim.components = components

            # Step 5: Evaluate claim
            verdict, confidence, reasoning = retry_function(self.claim_evaluator, claim)
            # Set claim attributes
            claim.verdict, claim.confidence, claim.reasoning = verdict, confidence, reasoning

        # If multiple claims, do verdict for entire statement
        if len(claims) > 1:
            # Step 6: Evaluate overall statement
            if VERBOSE: print_header("Overall Statement Evaluation", level=1, decorator='=')
            overall_verdict, overall_confidence, overall_reasoning = retry_function(self.overall_statement_evaluator, statement, claims)
            if VERBOSE:
                print_header(f"Overall Verdict: {colored(overall_verdict, 'green')}", level=2)
                print_header(f"Overall Confidence: {colored(str(overall_confidence), 'yellow')}", level=2)
                print_header(f"Overall Reasoning: {colored(overall_reasoning, 'cyan')}", level=2)
        else: 
            overall_verdict = claims[0].verdict
            overall_confidence = claims[0].confidence
            overall_reasoning = claims[0].reasoning

        if VERBOSE:
            print_header("Breakdown of Claims and Components", level=0, decorator='=')
            for i, claim in enumerate(claims, 1):
                print_header(f"Claim {i}", level=1)
                print_header("Text: " + colored(claim.text, 'white'), level=2)
                print_header("Verdict: " + colored(claim.verdict, 'green'), level=2)
                print_header("Confidence: " + colored(str(claim.confidence), 'yellow'), level=2)
                print_header("Reasoning: " + colored(claim.reasoning, 'cyan'), level=2)
                
                for j, component in enumerate(claim.components, 1):
                    print_header(f"Component {j}", level=2)
                    print_header("Question: " + colored(component.question, 'yellow'), level=3)
                    print_header("Answer: " + colored(component.answer.text, 'green'), level=3)
                    if component.answer.citations:
                        print_header("Citations: ", level=3)
                        for k, citation in enumerate(component.answer.citations, 1):
                            print_header(f"[{k}] " + colored(citation.snippet, 'yellow'), level=4)
                            print_header(f"Source: {citation.source_title} ({citation.source_url})", level=4)
                    else:
                        print_header("No explicit citations made by LM, but relevant documents used to synthesize answer: ")
                        unique_sources = {doc.metadata['url']: doc.metadata['title'] for doc in component.answer.retrieved_docs}
                        for url, title in unique_sources.items():
                            print_header(f"Title: {colored(title, 'yellow')}, URL: {colored(url, 'cyan')}", level=4)


        if VERBOSE:
            # Print final result
            print("\nFinal Fact-Check Result:")
            print_header(f"Statement: {colored(statement, 'white')}", level=1)
            print_header(f"Overall Verdict: {colored(overall_verdict, 'green')}", level=1)
            print_header(f"Overall Confidence: {colored(str(overall_confidence), 'yellow')}", level=1)
            print_header(f"Overall Reasoning: {colored(overall_reasoning, 'cyan')}", level=1)

        return overall_verdict, overall_confidence, overall_reasoning, claims

# Constants for whole pipeline
VERBOSE = True # Print intermediate results
INTERACTIVE = False # Allow the user to provide feedback

# Constants for Search Provider
NUM_SEARCH_RESULTS = 10 # Number of search results to retrieve
SCRAPE_TIMEOUT = 5 # Timeout for scraping a webpage (in seconds)

# Constants for Retrieval (Vector DB + BM25)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USE_BM25 = True # Use BM25 for retrieval (in addition to cosine similarity)
BM25_WEIGHT = 0.5 # Weight for BM25 in the hybrid retrieval

# Example usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv('../.env')

    # Initialize DSPy
    # lm = dspy.LM('gemini/gemini-1.5-flash', api_key=os.getenv('GOOGLE_GEMINI_API_KEY'), cache=False)
    # lm = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='', cache=False)
    lm = dspy.LM('ollama_chat/deepseek-r1:7b', api_base='http://localhost:11434', api_key='', cache=False)
    dspy.settings.configure(lm=lm)

    # Predefined knowledge base for the statements (allows for using your own documents instead of/in addition to web search)
    source_doc = """
    OpenAI o1 is a generative pre-trained transformer (GPT). A preview of o1 was released by OpenAI on September 12, 2024. o1 spends time "thinking" before it answers, making it better at complex reasoning tasks, science and programming than GPT-4o.[1] The full version was released to ChatGPT users on December 5, 2024.

    Capabilities
    According to OpenAI, o1 has been trained using a new optimization algorithm and a dataset specifically tailored to it; while also meshing in reinforcement learning into its training.[7] OpenAI described o1 as a complement to GPT-4o rather than a successor.[10][11]

    o1 spends additional time thinking (generating a chain of thought) before generating an answer, which makes it better for complex reasoning tasks, particularly in science and mathematics.[1] Compared to previous models, o1 has been trained to generate long "chains of thought" before returning a final answer.[12][13] According to Mira Murati, this ability to think before responding represents a new, additional paradigm, which is improving model outputs by spending more computing power when generating the answer, whereas the model scaling paradigm improves outputs by increasing the model size, training data and training compute power.[10] OpenAI's test results suggest a correlation between accuracy and the logarithm of the amount of compute spent thinking before answering.[13][12]

    o1-preview performed approximately at a PhD level on benchmark tests related to physics, chemistry, and biology. On the American Invitational Mathematics Examination, it solved 83% (12.5/15) of the problems, compared to 13% (1.8/15) for GPT-4o. It also ranked in the 89th percentile in Codeforces coding competitions.[14] o1-mini is faster and 80% cheaper than o1-preview. It is particularly suitable for programming and STEM-related tasks, but does not have the same "broad world knowledge" as o1-preview.[15]

    OpenAI noted that o1's reasoning capabilities make it better at adhering to safety rules provided in the prompt's context window. OpenAI reported that during a test, one instance of o1-preview exploited a misconfiguration to succeed at a task that should have been infeasible due to a bug.[16][17] OpenAI also granted early access to the UK and US AI Safety Institutes for research, evaluation, and testing. According to OpenAI's assessments, o1-preview and o1-mini crossed into "medium risk" in CBRN (biological, chemical, radiological, and nuclear) weapons. Dan Hendrycks wrote that "The model already outperforms PhD scientists most of the time on answering questions related to bioweapons." He suggested that these concerning capabilities will continue to increase.[18]
    """

    pipeline = FactCheckPipeline(
        model_name=lm,
        embedding_model=EMBEDDING_MODEL,
        search_provider=SearchProvider(provider="duckduckgo"),
        # context=[Document(content=source_doc, metadata={"title": "OpenAI o1", "url": "https://en.wikipedia.org/wiki/OpenAI_o1"})],
        retriever_k=5,
    ) 
    # statement = "In New York, there are no barriers to law enforcement to work with the federal government on immigration laws, and there are 100 crimes where migrants can be handed over."
    # statement = "OpenAI o1 can perform at a PhD level in physics."

    # statement = """"Crime is down in Venezuela by 67% because they're taking their gangs and their criminals and depositing them very nicely into the United States.”"""
    # verdict, confidence, reasoning, claims = pipeline.fact_check(statement, context="Source: Donald Trump, Date: 2024-04-02")
   
    # statement = """"Support for Roe is higher today in America than it has ever been.”"""
    # verdict, confidence, reasoning, claims = pipeline.fact_check(statement, context="Source: Joe Biden, Date: April 8, 2024")

    statement = """"The National Guard in the HISTORY of its life, gets called in AFTER a disaster, not BEFORE something happens.”"""
    verdict, confidence, reasoning, claims = pipeline.fact_check(
        statement=statement, 
        context=f"Statement Originator: Instagram posts, Date Claim Was Made: April 02, 2024"
    )