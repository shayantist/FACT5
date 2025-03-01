import os
import json
import requests
import itertools
from termcolor import colored
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Dict, Literal, Optional, Tuple
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
    index: int # index of the citation in the answer
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

        filtered_results = self._filter_and_rank_results(results)

        if len(filtered_results) == 0:
            print_header(f"No relevant sources found for query: {query}", level=1)
            if '\"' in query:
                query = query.replace('\"', '')
                print_header(f"Retrying with less restrictive quotes: {query}", level=1)
                return self.search(query, num_results=NUM_SEARCH_RESULTS)
            else:
                raise Exception(f"No relevant sources found for query: {query}")

        return filtered_results

    def _serper_search(self, query: str, num_results: int) -> List[SearchResult]:
        """
        use serper API for the given query and return search results
        *** params ***
        query: search query
        num_results: number of search results to return

        *** returns ***
        List of SearchResult objects
        """
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
        """
        search DuckDuckGo for the given query and return search results
        *** params ***
        query: search query
        num_results: number of search results to return

        *** returns ***
        List of SearchResult objects
        """
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
        
        # Remove low-quality results
        filtered_results = [
            result for result in unique_results
            if not self._is_blacklisted_source(result.url) and
            len(result.excerpt) > 50  # Minimum snippet length
        ]
        
        # Sort by reliability (TODO: implement more sophisticated ranking)
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
            'politifact.com', # FACT5 dataset source
            'snopes.com', # FACT5 dataset source
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
        self.documents = [] # TODO: turn into set/dict?
        self.metadata = [] # TODO: combine metadata with documents
        
        if use_bm25: self.bm25 = None
            
    def add_documents(self, documents: List[str], metadata: List[Dict]):
        if len(documents) == 0: return

        # Split documents into chunks in parallel
        with ThreadPoolExecutor() as executor:
            chunked_docs = list(
                executor.map(
                    chunk_text, 
                    documents, 
                    itertools.repeat(self.max_chunk_size), 
                    itertools.repeat(self.max_chunk_overlap)
                )
            )

        # Make sure no duplicate chunks are added, TODO: this is inefficient
        documents_set = set(self.documents)
        chunked_docs_deduped, metadata_deduped = [], []
        for chunks, meta in zip(chunked_docs, metadata):
            for chunk in chunks:
                if chunk not in documents_set:
                    chunked_docs_deduped.append(chunk)
                    metadata_deduped.append(meta)
            
        if len(chunked_docs_deduped) == 0:
            print(f"No non-duplicate chunks found. Skipping...")
            return

        # # Flatten chunks and duplicate metadata
        # all_chunks = []
        # all_metadata = []
        # for chunks, meta in zip(chunked_docs_deduped, metadata_deduped):
        #     all_chunks.extend(chunks)
        #     all_metadata.extend([meta] * len(chunks))

        # Encode chunks
        embeddings = self.encoder.encode(
            chunked_docs_deduped,
            convert_to_tensor=True,
        )
        
        # Initialize or update FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.cpu().numpy())
        self.documents.extend(chunked_docs_deduped)
        self.metadata.extend(metadata_deduped)
        
        # Initialize BM25 index
        if self.use_bm25:
            self.bm25 = BM25Okapi(self.documents)
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        # Get FAISS scores
        query_embedding = self.encoder.encode([query])[0].reshape(1, -1) # Reshape to 1D array
        
        # Error handling: if index is not initialized, raise an error
        if self.index is None:
            raise ValueError("Index is not initialized. Please add documents first.")
        
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
        """Clear the vector store."""
        self.index = None
        self.documents = []
        self.metadata = []

class ClaimExtractorSignature(dspy.Signature):
    """Extract specific claims from the given statement.
    1. Split the statement into multiple claims, but if the statement is atomic already (has one claim), return a list with just that claim.
    2. If context is included (e.g., time, location, source/speaker who made the statement, etc.), include the context in each claim to help verify it. Do not make up a context if it is not present in the text.
    3. Consider the source (e.g. name of the speaker, organization, etc.) and date of the statement if given in the context, and include them in each claim. 
    4. Each claim should be independent of each other and not refer to other claims.
    5. ALWAYS extract claims regardless of the content (fact, opinion, etc.).
    """
    statement = dspy.InputField(desc="Statement to extract claims from including any relevant context (e.g. source, speaker, date, etc.)")
    claims = dspy.OutputField(desc="""JSON objects of claims with the following schema: [{
        "text": string, # claim text
    }]""")

class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimExtractorSignature)

    def forward(self, statement: str) -> List[Claim]:
        # Extract claims with LLM
        try:
            result = self.extract(statement=statement)
            claims = json_repair.loads(result["claims"])
            claims = [Claim(text=claim.get("text", "")) for claim in claims]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(statement)
        
        # Error handling for non-factual claims (e.g., opinion)
        if len(claims) == 0:
            raise Exception(f"Failed to extract claim: {result.reasoning}") 
        
        return claims

class QuestionGeneratorSignature(dspy.Signature):
    """Break down the given claim derived from the original statement to generate independent questions and search queries to answer it. Remember that you are evaluating the truthfulness of the statement itself, not whether the statement was made, who it was made by, or when it was made. Be as specific and concise as possible, try to minimize the number of questions and search queries while still being comprehensive to verify the claim."""
    statement = dspy.InputField(desc="Original statement")
    claim = dspy.InputField(desc="Claim derived from the original statement to decompose into components (questions + search queries)")
    questions = dspy.OutputField(desc="""JSON objects of questions and search queries with the following schema: [{
        "question": string, # question text (e.g. "What was the GDP growth rate during the Trump administration?")
        "search_queries": [string], # independent search queries used to answer the question, be as specific as possible, no placeholders, avoid specific sites, ~1-2 queries is ideal
    }]""")
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
        
        return [ClaimComponent(**component) for component in data['questions']]

class QueryRefinerSignature(dspy.Signature):
    """Given the question and feeback provided, refine the given search queries to be more specific and concise to adequately answer the question."""
    question = dspy.InputField(desc="Question to answer")
    original_search_queries = dspy.InputField(desc="Original search queries which did not yield enough relevant information to answer the question")
    feedback = dspy.InputField(desc="Feedback on the search queries considering the question and the retrieved documents, containing information on what is missing and what is redundant")
    refined_search_queries = dspy.OutputField(desc="""JSON objects of refined search queries with the following schema: [{
        "search_query": string, # refined search query
    }]""")  
    
class QueryRefiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(QueryRefinerSignature)

    def forward(self, question: str, search_queries: List[str], feedback: str) -> List[str]:
        # Extract queries with LLM
        try:
            result = self.refine(question=question, original_search_queries=search_queries, feedback=feedback)
            queries = json_repair.loads(result["refined_search_queries"])
            queries = [query.get("search_query", "") for query in queries]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(question=question, original_search_queries=search_queries, feedback=feedback)
        
        # Error handling for non-factual claims (e.g., opinion)
        if len(queries) == 0:
            raise Exception(f"Failed to refine queries: {result.reasoning}") 
        
        return queries
    
class QuestionRefinerSignature(dspy.Signature):
    """Given the feeback provided, a claim, and the original statement the claim was derived from, refine the given questions and search queries to be more specific and concise to verify the claim with respect to the original statement."""
    statement = dspy.InputField(desc="Original statement")
    claim = dspy.InputField(desc="Claim derived from the original statement to decompose into components (questions + search queries)")
    original_questions_and_queries = dspy.InputField(desc="Original questions and search queries derived from the claim")
    feedback = dspy.InputField(desc="Feedback on the questions and search queries considering the claim and the retrieved documents, containing information on what is missing and what is redundant")
    new_questions_and_queries = dspy.OutputField(desc="""JSON objects of questions and search queries with the following schema: [{
        "question": string, # question text (e.g. "What was the GDP growth rate during the Trump administration?")
        "search_queries": [string], # independent search queries used to answer the question, be as specific as possible, no placeholders, avoid specific sites, ~1-2 queries is ideal
    }]""")

class QuestionRefiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(QuestionRefinerSignature)

    def forward(self, statement: str, claim: Claim, original_questions_and_queries: List[ClaimComponent], feedback: str) -> List[ClaimComponent]:
        # Extract original questions and search queries as JSON
        questions_and_queries_dict = [
            {
                "question": c.question,
                "search_queries": c.search_queries
            }
            for c in original_questions_and_queries
        ]

        # Extract queries with LLM
        try:
            result = self.refine(statement=statement, claim=claim, original_questions_and_queries=questions_and_queries_dict, feedback=feedback)
            data = json_repair.loads(str(result["new_questions_and_queries"]))
            
            # Error handling: if data is already a list, convert to dict
            if isinstance(data, list): data = {"questions": data}
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e} \nResponse: {result} \nRegenerating...")
            return self.forward(statement=statement, claim=claim, original_questions_and_queries=original_questions_and_queries, feedback=feedback)
        
        return [ClaimComponent(**component) for component in data['questions']]

## ANSWER SYNTHESIZER ##
class AnswerSynthesizerSignature(dspy.Signature):
    """Synthesize an answer based on retrieved documents with inline citations."""
    question: str = dspy.InputField(desc="Question to answer")
    search_queries = dspy.InputField(desc="Search queries used to find relevant information")
    documents = dspy.InputField(desc="Retrieved documents relevant to the question")
    has_sufficient_info: bool = dspy.OutputField(desc="Whether there was enough information to generate answer")
    citations = dspy.OutputField(desc="""JSON objects containing citations used to construct the answer with the following schema: [{
        "citation_id": int, # index of the citation in the answer above, should be unique and sequential (1, 2, ..., n)
        "document_id": int, # index of the document used from the documents provided above
        "snippet": string, # exact quote(s) from document content used to answer the question
        "relevance_score": float # relevance score of the citation to the question
    }]""")
    answer = dspy.OutputField(desc="Answer to the question with inline citations where the number in the brackets is corresponds to the 'citation_id' from above (e.g., 'The wage gap was shrinking [1]')")

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(AnswerSynthesizerSignature)

    def forward(self, component: ClaimComponent, documents: List[Document]) -> Tuple[Answer, bool]:
        # Only pass the necessary document metadata to the LLM to reduce input tokens
        documents_to_pass = [{ 
            "document_id": i,
            "document_title": doc.metadata.get("title", ""),
            "document_content": doc.content,
            "document_source": doc.metadata.get("source", "")
        } for i, doc in enumerate(documents, 1)]

        # Synthesize answer with LLM
        result = self.synthesize(
            question=component.question,
            search_queries=component.search_queries,
            documents=documents_to_pass,
        )
        
        # Parse citations and link back to documents
        citations = json_repair.loads(str(result["citations"]))

        citations = [Citation(
            index=c.get("citation_id", None),
            snippet=c.get("snippet", c.get("content", "")), 
            relevance_score=c.get("relevance_score", None),
            source_title=documents[c["document_id"]-1].metadata.get("title", "") if c.get("document_id", None) is not None else "",
            source_url=documents[c["document_id"]-1].metadata.get("url", "") if c.get("document_id", None) is not None else ""
        ) if isinstance(c, dict) else None for c in citations]

        answer = Answer(
            text=result["answer"],
            retrieved_docs=documents,
            citations=citations
        )

        return answer, result["has_sufficient_info"]

## CLAIM EVALUATOR ##
VERDICTS = Literal[
    "TRUE", "MOSTLY TRUE", "HALF TRUE", 
    "MOSTLY FALSE", "FALSE", "UNVERIFIABLE"
]

class ClaimEvaluatorSignature(dspy.Signature):
    """Evaluate a claim's truthfulness based on questions and answers."""
    claim = dspy.InputField(desc="Claim to evaluate")
    qa_pairs = dspy.InputField(desc="Question-answer pairs with citations")
    verdict: VERDICTS = dspy.OutputField(desc="Verdict for the claim")
    confidence: float = dspy.OutputField(desc="Confidence score for the verdict")
    # TODO: inline citations for each question-answer pair
        # "evidence_analysis": [
        #     {{
        #         "question": string,
        #         "answer": string,
        #         "contribution": string
        #     }}

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
        
        return result.verdict, result.confidence, result.reasoning

## OVERALL STATEMENT EVALUATOR (SET OF CLAIMS) ##
class OverallStatementEvaluatorSignature(dspy.Signature):
    """Calculate ONE overall verdict for the entire statement based on the verdicts of each atomic claim. Remember that you are evaluating the truthfulness of the statement itself, not whether the statement was made, who it was made by, or when it was made."""
    statement = dspy.InputField(desc="Overall statement to evaluate")
    claims = dspy.InputField(desc="List of evaluated atomic claims derived from the statement, and associated question-answer pairs")
    overall_verdict: VERDICTS = dspy.OutputField(desc="Single verdict for the overall statement (do not treat each claim as of equal importance, weigh each claim depending on how much the statement depends on it)")
    confidence: float = dspy.OutputField(desc="Confidence score for the overall verdict")

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
                
        return result.overall_verdict, result.confidence, result.reasoning

class FactCheckPipeline:
    def __init__(
        self,
        model_name: str,
        embedding_model: str,
        retriever_k: int = 10,
        search_provider: SearchProvider = None,
        context: List[Document] = None,
        self_correct_per_claim: bool = False, # Pipeline will self-correct if it detects that the claim is unverifiable
        self_correct_per_answer: bool = False, # Pipeline will self-correct if it detects that the answer doesn't have enough info
        num_retries_per_answer: int = 3,
        num_retries_per_claim: int = 3,
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

        # Initialize self-correcting components
        self.self_correct_per_claim = self_correct_per_claim
        self.self_correct_per_answer = self_correct_per_answer
        self.max_retries = num_retries_per_answer if self_correct_per_answer else num_retries_per_claim
        self.query_refiner = QueryRefiner()
        self.question_refiner = QuestionRefiner()

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

        # If verbose, print extracted claims
        if VERBOSE:
            print_header(f"Extracted Claims ({len(claims)}): ", level=1)
            for i, claim in enumerate(claims, 1):
                print_header(f"{i}. {colored(claim.text, 'white')}", level=2, decorator='')
                # print_header(f"  Reasoning: {claim['reasoning']}", level=3)

        # TODO: allow user to provide feedback to regenerate claims
        # if INTERACTIVE:
        #     while True:
        #         try:
        #             feedback = input("Are these claims correctly extracted? (yes/no): ").lower()
        #             if feedback == "yes":
        #                 break
        #             elif feedback == "no":
        #                 feedback = input("Please provide feedback on what's wrong: ")
        #                 # Regenerate claims with feedback (TODO: implement history)
        #                 statement = f"{statement} {feedback}"
        #                 return self.forward(statement=statement)
        #         except KeyboardInterrupt:
        #             print("\nProcess interrupted. Exiting...")
        #             exit(0)

        for claim_i, claim in enumerate(claims, 1):
            # Step 2: Decompose claim into components (questions and search queries)
            if VERBOSE: print_header(f"Question Generation [{claim_i}/{len(claims)}]", level=2, decorator='=')
            components = retry_function(self.question_generator, statement, claim) # List of ClaimComponent objects

            # Initialize claim verification flag and retries
            claim_verified = False
            retries_per_claim = 0
            while not claim_verified and retries_per_claim < self.max_retries:
                # If verbose, print generated components
                if VERBOSE:
                    print_header(f"Decomposed Components (Questions + Search Queries) ({len(components)}):", level=3)
                    for i, component in enumerate(components, 1):
                        print_header(f"{i}. Question: {colored(component.question, 'yellow')}", level=4)
                        print_header(f"   Search Queries: {colored(component.search_queries, 'yellow')}", level=4)

                # TODO: allow user to provide feedback to regenerate components
                # if INTERACTIVE:
                #     while True:
                #         feedback = input("Are these components correct? (yes/no): ").lower()
                #         if feedback == "yes":
                #             break
                #         elif feedback == "no":
                #             feedback = input("Please provide feedback on what's wrong: ")
                #             # Regenerate components with feedback (TODO: implement history)
                #             result = self.forward(claim=f"{claim.text} {feedback}")

                for component_i, component in enumerate(components, 1):
                    # Keep asking until answer is synthesized or max retries reached
                    answered = False 
                    retries_per_answer = 0

                    while not answered and retries_per_answer < self.max_retries:
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

                                if VERBOSE:
                                    print_header(f"Retrieved {len(search_results)} sources from the web:", level=4)
                                    for i, result in enumerate(search_results, 1):
                                        print_header(f"[{i}] {colored(result.title, 'yellow')}", level=5)
                                        print_header(f"URL: {colored(result.url, 'cyan')}", level=5)
                                        print_header(f"Excerpt: {colored(result.excerpt[:150], 'magenta')}", level=5)
                                
                                # Save documents (search results) to vector DB
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

                                # TODO: Allow user to provide feedback on retrieved documents
                                # if INTERACTIVE:
                                #     while True:
                                #         feedback = input("Are these sources relevant? (yes/no): ").lower()
                                #         if feedback == "yes":
                                #             break
                                #         elif feedback == "no":
                                #             feedback = input("Please provide feedback on what's wrong: ")
                                #             # Append feedback to query
                                #             query += f" {feedback}" 
                                #             continue
                                #     else: 
                                #         web_search_approved = True

                            # Get ONLY **relevant** documents (search results)
                            relevant_docs.extend(self.retriever.retrieve(query, k=self.retriever_k))
                        
                        if VERBOSE: 
                            print_header(f"Retrieved {len(relevant_docs)} documents from internal knowledge base:", level=4, decorator='=')
                            # unique_sources = {doc.metadata['url']: doc.metadata['title'] for doc in relevant_docs}
                            # for i, (url, title) in enumerate(unique_sources.items(), 1):
                            for i, doc in enumerate(relevant_docs, 1):
                                print_header(f"[{i}] {colored(doc.metadata['title'], 'yellow')}", level=5)
                                print_header(f"URL: {colored(doc.metadata['url'], 'cyan')}", level=5)
                                print_header(f"Snippet: {colored(doc.content[:150], 'magenta')}...", level=5)

                        # Step 4: Synthesize answer given relevant documents
                        if VERBOSE: 
                            print_header(f"Synthesizing Answer [{component_i}/{len(components)}]", level=3, decorator='=')
                            print_header("Question: " + colored(component.question, 'yellow'), level=4)
                            # print_header("Search Queries: " + colored(component.search_queries, 'yellow'), level=4)

                        # answer = retry_function(self.answer_synthesizer, component, documents=relevant_docs)

                        # Try to synthesize answer
                        answer, has_sufficient_info = self.answer_synthesizer(component, documents=relevant_docs)
                        
                        if VERBOSE:
                            print_header(f"Answer: {colored(answer.text, 'green')}", level=4)
                            if answer.citations:
                                print_header("Citations: ", level=4)
                                for i, citation in enumerate(answer.citations, 1):
                                    print_header(f"[{i}] " + colored(citation.snippet, 'yellow'), level=5)
                                    print_header(f"Source: {citation.source_title} ({citation.source_url})", level=5)

                        # If self-correct is enabled AND not enough information, generate new search queries and redo step 3
                        if self.self_correct_per_answer and not has_sufficient_info:
                            print_header(f"[Attempt {retries_per_answer + 1}/{self.max_retries}] Not enough information to synthesize answer, generating new search queries...", color='red', level=4)
                            relevant_docs = [] # Reset relevant documents
                            component.search_queries = self.query_refiner(
                                question=component.question, 
                                search_queries=component.search_queries, 
                                feedback=answer.text
                            )
                            retries_per_answer += 1
                            continue
                        else:
                            answered = True
                        
                        # TODO: allow user to provide feedback to regenerate answer
                        # if INTERACTIVE:
                        #     while True:
                        #         feedback = input("Is this answer correct? (yes/no): ").lower()
                        #         if feedback == "yes":
                        #             break
                        #         elif feedback == "no":
                        #             feedback = input("Please provide feedback on what's wrong: ")
                        #             # Regenerate answer with feedback (TODO: implement history)
                        #             result = self.forward(component=f"{component.question} {feedback}")    

                    component.answer = answer # Set answer to the question

                    # Set claim components (questions and answers)
                    claim.components = components

                # Step 5: Evaluate claim
                verdict, confidence, reasoning = retry_function(self.claim_evaluator, claim)

                if VERBOSE or INTERACTIVE:
                    print_header("Claim Evaluation", level=2, decorator='=')
                    print_header(f"Claim: {colored(claim.text, 'white')}", level=3)
                    print_header(f"Verdict: {colored(verdict, 'green')}", level=3)
                    print_header(f"Confidence: {colored(str(confidence), 'yellow')}", level=3)
                    print_header(f"Reasoning: {colored(reasoning, 'green')}", level=3)
                    
                    for j, component in enumerate(claim.components, 1):
                        print_header(f"Component {j}", level=3, decorator='=')
                        print_header("Question: " + colored(component.question, 'yellow'), level=4)
                        print_header("Answer: " + colored(component.answer.text, 'green'), level=4)
                        if component.answer.citations:
                            print_header("Citations: ", level=4)
                            for k, citation in enumerate(component.answer.citations, 1):
                                print_header(f"[{k}] " + colored(citation.snippet, 'magenta'), level=5)
                                print_header(f"Source: {colored(citation.source_title, 'yellow')} ({colored(citation.source_url, 'cyan')})", level=5)
                        else:
                            print_header("No explicit citations made by LM, but relevant documents used to synthesize answer: ")
                            for k, doc in enumerate(component.answer.retrieved_docs, 1):
                                print_header(f"[{k}] {colored(doc.content[:150], 'magenta')}...", level=5)
                                print_header(f"Source: {colored(doc.metadata['title'], 'yellow')} ({colored(doc.metadata['url'], 'cyan')})", level=5)

                # TODO: allow user to provide feedback to regenerate verdict for claim
                # if INTERACTIVE:
                #     while True:
                #         feedback = input("Is this verdict correct? (yes/no): ").lower()
                #         if feedback == "yes":
                #             break

                # If self-correct is enabled AND claim is unverifiable, generate new questions and search queries and redo step 2
                if self.self_correct_per_claim and verdict == "UNVERIFIABLE":
                    print_header(f"[Attempt {retries_per_claim + 1}/{self.max_retries}] Claim is unverifiable, generating new questions and search queries...", color='red', level=4)
                    components = self.question_refiner(
                        statement=statement, 
                        claim=claim, 
                        original_questions_and_queries=claim.components, 
                        feedback=answer.text
                    )
                    retries_per_claim += 1
                    continue
                else:
                    claim_verified = True

            # Set claim attributes
            claim.verdict, claim.confidence, claim.reasoning = verdict, confidence, reasoning

        # If multiple claims, do verdict for entire statement
        if len(claims) > 1:
            # Step 6: Evaluate overall statement
            if VERBOSE: print_header("Overall Statement Evaluation", level=1, decorator='=')
            overall_verdict, overall_confidence, overall_reasoning = retry_function(self.overall_statement_evaluator, statement, claims)

            # TODO: allow user to provide feedback to regenerate overall verdict
            # if INTERACTIVE:
            #     while True:
            #         feedback = input("Is this verdict correct? (yes/no): ").lower()
            #         if feedback == "yes":
            #             break
            #         elif feedback == "no":
            #             feedback = input("Please provide feedback on what's wrong: ")
            #             # Regenerate overall verdict with feedback (TODO: implement history)
            #             result = self.forward(statement=f"{statement} {feedback}")    

            if VERBOSE:
                print_header(f"Overall Verdict: {colored(overall_verdict, 'green')}", level=2)
                print_header(f"Overall Confidence: {colored(str(overall_confidence), 'yellow')}", level=2)
                print_header(f"Overall Reasoning: {colored(overall_reasoning, 'green')}", level=2)
        else: 
            overall_verdict = claims[0].verdict
            overall_confidence = claims[0].confidence
            overall_reasoning = claims[0].reasoning

        if VERBOSE:
            print_header("Breakdown of Claims and Components", level=0, decorator='=')
            for i, claim in enumerate(claims, 1):
                print_header(f"Claim {i}", level=1, decorator='=')
                print_header("Text: " + colored(claim.text, 'yellow'), level=2)
                print_header("Verdict: " + colored(claim.verdict, 'green'), level=2)
                print_header("Confidence: " + colored(str(claim.confidence), 'yellow'), level=2)
                print_header("Reasoning: " + colored(claim.reasoning, 'green'), level=2)
                
                for j, component in enumerate(claim.components, 1):
                    print_header(f"Component {j}", level=2, decorator='=')
                    print_header("Question: " + colored(component.question, 'yellow'), level=3)
                    print_header("Answer: " + colored(component.answer.text, 'green'), level=3)
                    if component.answer.citations:
                        print_header("Citations: ", level=3)
                        for k, citation in enumerate(component.answer.citations, 1):
                            print_header(f"[{k}] " + colored(citation.snippet, 'magenta'), level=4)
                            print_header(f"Source: {colored(citation.source_title, 'yellow')} ({colored(citation.source_url, 'cyan')})", level=4)
                    else:
                        print_header("No explicit citations made by LM, but relevant documents used to synthesize answer: ")
                        for k, doc in enumerate(component.answer.retrieved_docs, 1):
                            print_header(f"[{k}] {colored(doc.content[:150], 'magenta')}...", level=4)
                            print_header(f"Source: {colored(doc.metadata['title'], 'yellow')} ({colored(doc.metadata['url'], 'cyan')})", level=4)

        if VERBOSE:
            # Print final result
            print("\nFinal Fact-Check Result:")
            print_header(f"Statement: {colored(statement, 'yellow')}", level=1)
            print_header(f"Overall Verdict: {colored(overall_verdict, 'green')}", level=1)
            print_header(f"Overall Confidence: {colored(str(overall_confidence), 'yellow')}", level=1)
            print_header(f"Overall Reasoning: {colored(overall_reasoning, 'green')}", level=1)

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
    dotenv.load_dotenv('../.env', override=True)

    # Initialize DSPy
    lm = dspy.LM('gemini/gemini-1.5-flash', api_key=os.getenv('GOOGLE_GEMINI_API_KEY'), cache=False)
    # lm = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='', cache=False)
    dspy.settings.configure(lm=lm)

    # Predefined knowledge base for the statements (allows for using your own documents instead of/in addition to web search)
    source_docs = [
        {
            "title": "OpenAI o1",
            "url": "https://en.wikipedia.org/wiki/OpenAI_o1",
            "content": """
                OpenAI o1 is a generative pre-trained transformer (GPT). A preview of o1 was released by OpenAI on September 12, 2024. o1 spends time "thinking" before it answers, making it better at complex reasoning tasks, science and programming than GPT-4o.[1] The full version was released to ChatGPT users on December 5, 2024.

                Capabilities
                According to OpenAI, o1 has been trained using a new optimization algorithm and a dataset specifically tailored to it; while also meshing in reinforcement learning into its training.[7] OpenAI described o1 as a complement to GPT-4o rather than a successor.[10][11]

                o1 spends additional time thinking (generating a chain of thought) before generating an answer, which makes it better for complex reasoning tasks, particularly in science and mathematics.[1] Compared to previous models, o1 has been trained to generate long "chains of thought" before returning a final answer.[12][13] According to Mira Murati, this ability to think before responding represents a new, additional paradigm, which is improving model outputs by spending more computing power when generating the answer, whereas the model scaling paradigm improves outputs by increasing the model size, training data and training compute power.[10] OpenAI's test results suggest a correlation between accuracy and the logarithm of the amount of compute spent thinking before answering.[13][12]

                o1-preview performed approximately at a PhD level on benchmark tests related to physics, chemistry, and biology. On the American Invitational Mathematics Examination, it solved 83% (12.5/15) of the problems, compared to 13% (1.8/15) for GPT-4o. It also ranked in the 89th percentile in Codeforces coding competitions.[14] o1-mini is faster and 80% cheaper than o1-preview. It is particularly suitable for programming and STEM-related tasks, but does not have the same "broad world knowledge" as o1-preview.[15]

                OpenAI noted that o1's reasoning capabilities make it better at adhering to safety rules provided in the prompt's context window. OpenAI reported that during a test, one instance of o1-preview exploited a misconfiguration to succeed at a task that should have been infeasible due to a bug.[16][17] OpenAI also granted early access to the UK and US AI Safety Institutes for research, evaluation, and testing. According to OpenAI's assessments, o1-preview and o1-mini crossed into "medium risk" in CBRN (biological, chemical, radiological, and nuclear) weapons. Dan Hendrycks wrote that "The model already outperforms PhD scientists most of the time on answering questions related to bioweapons." He suggested that these concerning capabilities will continue to increase.[18]"""
        },
        {
            "title": "Bangladesh",
            "url": "https://en.wikipedia.org/wiki/Bangladesh",
            "content": """Bangladesh is a country in South Asia. It gained independence from Pakistan in 1971."""
        }
    ]

    pipeline = FactCheckPipeline(
        model_name=lm,
        embedding_model=EMBEDDING_MODEL,
        search_provider=SearchProvider(provider="serper", api_key=os.getenv('SERPER_API_KEY')),
        # context=[Document(content=doc["content"], metadata={"title": doc["title"], "url": doc["url"]}) for doc in source_docs],
        retriever_k=5,
        self_correct_per_claim=True,
        self_correct_per_answer=True,
        # num_retries_per_answer=3,
        # num_retries_per_claim=3,
    )
  
    # statement = "o1 is better than GPT-4o"
    # statement = "Bangladesh is a country in South Asia that gained independence from India in 1971."
    statement = "Kamala Harris claimed on October 23, 2024, that 'As of today, we have cut the flow of immigration by over half."
    verdict, confidence, reasoning, claims = pipeline.fact_check(statement=statement)

    print()