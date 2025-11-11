"""
Q Chat Web Interface Backend - V6.3 HYBRID (FIXED + TIMESTAMPS)
Author: KISHORE
Date: 2025-01-06 18:00:00 UTC

CHANGES:
- ✅ Added timestamp to context format: {context:1(timestamp: ISO_TIME) {question:...} {answer:...}}
- ✅ Deduplication: Removes duplicate contexts, keeps most recent
- ✅ Enhanced prompt view: Shows original query + formatted prompt + raw output
"""

import os
import sys
import json
import asyncio
import tempfile
import re
import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# ============================================================================
# DISABLE TELEMETRY FIRST
# ============================================================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from unittest.mock import MagicMock
sys.modules['posthog'] = MagicMock()

# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings

from rich.console import Console

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "embeddinggemma:latest"
OLLAMA_URL = "http://localhost:11434"
DB_PATH = os.getenv("PERSIST_DIRECTORY", "./qchat_web_memory")
Q_CHAT_COMMAND = "q"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def strip_ansi_and_ascii_art(text: str) -> str:
    """Remove ANSI codes AND ASCII art completely"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    lines = text.split('\n')
    clean_lines = []
    
    skip_patterns = [
        '⢠', '⣶', '⠀', '⡿', '⣿',
        '╭', '╮', '╰', '╯', '│', '─', '━',
        '/help', 'ctrl +', 'Did you know?',
        'You are chatting with', 'To exit',
        'Allow this action?', '[y/n/t]',
        'All tools are now trusted',
        'Learn more at https://',
        'Agents can sometimes',
        '⋮', '●', '⬤', '🤖'
    ]
    
    for line in lines:
        if not any(pattern in line for pattern in skip_patterns):
            stripped = line.strip()
            if stripped and stripped not in ['>', '', '⋮', '●']:
                clean_lines.append(stripped)
    
    result = '\n'.join(clean_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def clean_for_inline(text: str, max_len: int = 100) -> str:
    """
    Clean text for inline single-line format:
    - Remove newlines
    - Remove extra whitespace
    - Escape special characters
    - Truncate to max length
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = text.replace('{', '(').replace('}', ')')
    
    if len(text) > max_len:
        text = text[:max_len] + "..."
    
    return text.strip()


def format_contexts_single_line(contexts: List[Dict]) -> str:
    """
    Format RAG contexts with timestamps:
    {context:1(timestamp: 2025-01-06T16:30:45Z) {question:...} {answer:...}}
    
    Args:
        contexts: List of context dicts with 'query', 'response', and 'timestamp' keys
    
    Returns:
        Single-line formatted string
    """
    if not contexts:
        return ""
    
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        question = ctx.get('query', '').strip()
        answer = ctx.get('response', '').strip()
        timestamp = ctx.get('timestamp', datetime.utcnow().isoformat())
        
        # Clean and truncate
        question = clean_for_inline(question, max_len=100)
        answer = clean_for_inline(answer, max_len=150)
        
        # Format with timestamp: {context:N(timestamp: ISO_TIME) {question:...} {answer:...}}
        context_str = f"{{context:{i}(timestamp: {timestamp}) {{question:{question}}} {{answer:{answer}}}}}"
        context_parts.append(context_str)
    
    return " ".join(context_parts)


def format_history_single_line(history: List[Dict]) -> str:
    """
    Format conversation history as: {history:1 {question:...} {answer:...}} {history:2 {question:...} {answer:...}}
    
    Args:
        history: List of exchange dicts with 'query' and 'response' keys
    
    Returns:
        Single-line formatted string
    """
    if not history:
        return ""
    
    history_parts = []
    
    for i, exchange in enumerate(history, 1):
        question = exchange.get('query', '').strip()
        answer = exchange.get('response', '').strip()
        
        # Clean and truncate
        question = clean_for_inline(question, max_len=80)
        answer = clean_for_inline(answer, max_len=120)
        
        # Format as {history:N {question:...} {answer:...}}
        history_str = f"{{history:{i} {{question:{question}}} {{answer:{answer}}}}}"
        history_parts.append(history_str)
    
    return " ".join(history_parts)


def build_single_line_prompt(
    user_query: str,
    contexts: List[Dict] = None,
    history: List[Dict] = None
) -> str:
    """
    Build complete single-line prompt with contexts and history.
    
    Format:
    {context:1(timestamp: ...) {question:...} {answer:...}} {current:{user query here}}
    
    Args:
        user_query: Current user question
        contexts: RAG contexts from vector DB (with timestamps)
        history: Conversation history from session
    
    Returns:
        Complete single-line prompt
    """
    parts = []
    
    # Add RAG contexts (with timestamps)
    if contexts:
        contexts_str = format_contexts_single_line(contexts[:2])  # Max 2 contexts
        if contexts_str:
            parts.append(contexts_str)
    
    # Add conversation history
    if history:
        history_str = format_history_single_line(history[-2:])  # Last 2 exchanges
        if history_str:
            parts.append(history_str)
    
    # Add current query
    clean_query = clean_for_inline(user_query, max_len=500)
    parts.append(f"{{current:{clean_query}}}")
    
    # Join everything with space
    full_prompt = " ".join(parts)
    
    # Final safety check - ensure truly single line
    full_prompt = full_prompt.replace('\n', ' ').replace('\r', ' ')
    full_prompt = re.sub(r'\s+', ' ', full_prompt)
    
    # Truncate if exceeds Q Chat limits
    max_total_length = 4000
    if len(full_prompt) > max_total_length:
        console.print(f"[yellow]⚠️  Truncating prompt from {len(full_prompt)} to {max_total_length} chars[/yellow]")
        full_prompt = full_prompt[:max_total_length] + "...}"
    
    console.print(f"[cyan]📝 Single-line prompt: {len(full_prompt)} chars[/cyan]")
    
    return full_prompt.strip()

# ============================================================================
# OLLAMA EMBEDDINGS
# ============================================================================

class OllamaEmbeddings:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url
        self._test_connection()
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Ollama connected[/green]")
        except Exception as e:
            console.print(f"[red]❌ Ollama error: {e}[/red]")
            raise
    
    def embed(self, text: str) -> List[float]:
        try:
            if len(text) > 8000:
                text = text[:8000]
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            return []
        except Exception as e:
            console.print(f"[red]Embedding error: {e}[/red]")
            return []

# ============================================================================
# Q CHAT EXECUTOR
# ============================================================================

class QChatExecutor:
    @staticmethod
    async def execute(
        prompt: str,
        contexts: List[Dict] = None,
        history: List[Dict] = None
    ) -> Dict[str, str]:
        """
        Execute Q Chat with single-line structured prompt.
        
        Format: {context:1(timestamp: ...) {question:...} {answer:...}} {current:...}
        
        Args:
            prompt: User query
            contexts: RAG contexts (optional, with timestamps)
            history: Conversation history (optional)
        
        Returns:
            Dict with 'clean', 'raw', and 'has_raw' keys
        """
        
        # Build single-line prompt
        formatted_prompt = build_single_line_prompt(prompt, contexts, history)
        
        # Log for debugging
        console.print(f"[dim]Formatted prompt preview:[/dim]")
        console.print(f"[dim]{formatted_prompt[:200]}...[/dim]")
        
        # Verify it's single line
        if '\n' in formatted_prompt or '\r' in formatted_prompt:
            console.print("[red]⚠️  ERROR: Multi-line detected! Forcing single-line...[/red]")
            formatted_prompt = formatted_prompt.replace('\n', ' ').replace('\r', ' ')
        
        # Write to temp file (single line, no trailing newline)
        with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8',
            newline=''
        ) as f:
            f.write(formatted_prompt)
            temp_file = f.name
        
        # Execute Q Chat
        command = f'{Q_CHAT_COMMAND} chat --trust-all-tools < "{temp_file}"'
        
        try:
            console.print(f"[yellow]🚀 Executing Q Chat...[/yellow]")
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    'TERM': 'dumb',
                    'NO_COLOR': '1'
                }
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600
            )
            
            try:
                os.unlink(temp_file)
            except:
                pass
            
            # Process output
            raw_output = ""
            clean_output = ""
            
            if stdout:
                raw_stdout = stdout.decode('utf-8', errors='ignore')
                raw_output = raw_stdout
                clean_output = strip_ansi_and_ascii_art(raw_stdout)
                
                if clean_output and len(clean_output) > 20:
                    console.print(f"[green]✅ stdout: {len(clean_output)} chars[/green]")
                    return {
                        'clean': clean_output,
                        'raw': raw_output,
                        'has_raw': len(raw_output) > len(clean_output) + 50
                    }
            
            if stderr:
                raw_stderr = stderr.decode('utf-8', errors='ignore')
                if not raw_output:
                    raw_output = raw_stderr
                clean_stderr = strip_ansi_and_ascii_art(raw_stderr)
                
                if clean_stderr and len(clean_stderr) > 50:
                    console.print(f"[yellow]✅ stderr: {len(clean_stderr)} chars[/yellow]")
                    return {
                        'clean': clean_stderr,
                        'raw': raw_output,
                        'has_raw': len(raw_output) > len(clean_stderr) + 50
                    }
            
            console.print("[yellow]⚠️  No substantial output captured[/yellow]")
            return {
                'clean': "✅ Command executed (no output captured)",
                'raw': raw_output if raw_output else "No raw output",
                'has_raw': False
            }
            
        except asyncio.TimeoutError:
            console.print("[red]❌ Timeout (10 minutes)[/red]")
            try:
                os.unlink(temp_file)
            except:
                pass
            return {
                'clean': "⚠️ Query timeout. Try simpler query or check Q Chat logs.",
                'raw': "Timeout error",
                'has_raw': False
            }
            
        except Exception as e:
            console.print(f"[red]❌ Execution error: {e}[/red]")
            import traceback
            traceback.print_exc()
            
            try:
                os.unlink(temp_file)
            except:
                pass
            
            return {
                'clean': f"❌ Error: {str(e)}",
                'raw': traceback.format_exc(),
                'has_raw': True
            }

# ============================================================================
# CONVERSATION SESSION TRACKER
# ============================================================================

class ConversationSession:
    """Tracks conversation history for context injection"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict] = []
        self.last_activity = time.time()
        self.max_history = 5
    
    def add_exchange(self, query: str, response: str):
        """Add Q&A to history"""
        self.history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.last_activity = time.time()
    
    def is_expired(self) -> bool:
        """Check if session expired (10 minutes)"""
        return time.time() - self.last_activity > 600


class SessionManager:
    """Manages conversation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
    
    def get_session(self, session_id: str) -> ConversationSession:
        """Get or create session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id)
        
        session = self.sessions[session_id]
        
        if session.is_expired():
            self.sessions[session_id] = ConversationSession(session_id)
            session = self.sessions[session_id]
        
        return session
    
    def close_session(self, session_id: str):
        """Close session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# ============================================================================
# RAG ENGINE WITH DEDUPLICATION
# ============================================================================

class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings()
        
        self.client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.interactions = self.client.get_or_create_collection(
            name="qchat_interactions",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.conversations = self.client.get_or_create_collection(
            name="qchat_conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        console.print(f"[cyan]📊 Interactions: {self.interactions.count()}[/cyan]")
        console.print(f"[cyan]💬 Conversations: {self.conversations.count()}[/cyan]")
    
    def deduplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """
        Remove duplicate contexts, keeping the most recent one.
        Contexts are considered duplicates if their queries are very similar (>95% similarity).
        
        Args:
            contexts: List of context dicts with 'query', 'response', 'timestamp', 'similarity'
        
        Returns:
            Deduplicated list sorted by relevance
        """
        if len(contexts) <= 1:
            return contexts
        
        # Group by similar questions
        groups = []
        
        for ctx in contexts:
            placed = False
            
            for group in groups:
                # Compare with first item in group (representative)
                rep_query = group[0]['query'].lower()
                ctx_query = ctx['query'].lower()
                
                # Simple similarity: check if queries are very similar
                # (You can use more sophisticated text similarity here)
                if self._text_similarity(rep_query, ctx_query) > 0.95:
                    group.append(ctx)
                    placed = True
                    break
            
            if not placed:
                groups.append([ctx])
        
        # Keep most recent from each group
        deduplicated = []
        for group in groups:
            # Sort by timestamp (most recent first)
            group_sorted = sorted(
                group,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            deduplicated.append(group_sorted[0])
        
        # Sort by relevance (similarity score)
        deduplicated.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        if len(contexts) != len(deduplicated):
            console.print(f"[yellow]🔄 Deduplicated: {len(contexts)} -> {len(deduplicated)} contexts[/yellow]")
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def search_context(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Search for relevant context with deduplication.
        Returns up to top_k deduplicated contexts.
        """
        if self.interactions.count() == 0:
            return []
        
        embedding = self.embeddings.embed(query)
        if not embedding:
            return []
        
        try:
            # Fetch more results initially to allow for deduplication
            n_results = min(top_k * 2, self.interactions.count())
            if n_results == 0:
                return []
            
            results = self.interactions.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            
            contexts = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    similarity = 1 - results['distances'][0][i]
                    
                    if similarity > 0.5:
                        full_content = results['documents'][0][i]
                        metadata = results['metadatas'][0][i]
                        
                        query_part = ""
                        response_part = ""
                        
                        if "Query:" in full_content and "Response:" in full_content:
                            parts = full_content.split("Response:", 1)
                            query_part = parts[0].replace("Query:", "").strip()
                            response_part = parts[1].split("Timestamp:")[0].strip()
                        
                        contexts.append({
                            'query': query_part,
                            'response': response_part[:300],
                            'timestamp': metadata.get('timestamp', datetime.utcnow().isoformat()),
                            'similarity': similarity
                        })
            
            # Deduplicate contexts
            contexts = self.deduplicate_contexts(contexts)
            
            # Return top_k after deduplication
            contexts = contexts[:top_k]
            
            console.print(f"[green]Found {len(contexts)} deduplicated contexts[/green]")
            return contexts
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []
    
    def store_interaction(self, query: str, response: str, raw_response: str, conversation_id: str):
        """Store interaction with timestamp in metadata"""
        clean_response = strip_ansi_and_ascii_art(response)
        timestamp = datetime.utcnow().isoformat()
        
        full_context = f"""Query: {query}
Response: {clean_response}
Timestamp: {timestamp}
User: KISHORE
Conversation: {conversation_id}"""
        
        embedding = self.embeddings.embed(full_context)
        if not embedding:
            return
        
        meta = {
            "timestamp": timestamp,  # Store timestamp in metadata
            "user": "KISHORE",
            "conversation_id": conversation_id,
            "query_preview": query[:100],
            "response_length": len(clean_response),
            "has_raw": len(raw_response) > len(clean_response) + 50,
            "raw_response": raw_response[:5000] if raw_response else ""
        }
        
        interaction_id = f"int_{datetime.utcnow().timestamp()}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.interactions.add(
                embeddings=[embedding],
                documents=[full_context],
                metadatas=[meta],
                ids=[interaction_id]
            )
            console.print(f"[green]✅ Stored interaction with timestamp: {timestamp}[/green]")
        except Exception as e:
            console.print(f"[red]Store error: {e}[/red]")
    
    def create_conversation(self, title: str = "New Conversation") -> str:
        """Create new conversation"""
        conv_id = f"conv_{uuid.uuid4().hex}"
        
        doc = f"Conversation: {title}\nCreated: {datetime.utcnow().isoformat()}\nUser: KISHORE"
        embedding = self.embeddings.embed(doc)
        
        try:
            self.conversations.add(
                embeddings=[embedding],
                documents=[doc],
                metadatas={
                    "conversation_id": conv_id,
                    "title": title,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "user": "KISHORE",
                    "message_count": 0
                },
                ids=[conv_id]
            )
            console.print(f"[cyan]📝 Created conversation: {conv_id}[/cyan]")
            return conv_id
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return conv_id
    
    def update_conversation(self, conversation_id: str, title: str = None, increment_count: bool = False):
        """Update conversation"""
        try:
            result = self.conversations.get(ids=[conversation_id])
            if not result['metadatas']:
                return
            
            meta = result['metadatas'][0]
            meta['updated_at'] = datetime.utcnow().isoformat()
            
            if title:
                meta['title'] = title
            
            if increment_count:
                meta['message_count'] = meta.get('message_count', 0) + 1
            
            self.conversations.update(
                ids=[conversation_id],
                metadatas=[meta]
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def get_conversations(self, limit: int = 50) -> List[Dict]:
        """Get conversation list"""
        try:
            results = self.conversations.get(limit=limit)
            
            conversations = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    meta = results['metadatas'][i]
                    conversations.append({
                        'id': results['ids'][i],
                        'title': meta.get('title', 'Untitled'),
                        'created_at': meta.get('created_at'),
                        'updated_at': meta.get('updated_at'),
                        'message_count': meta.get('message_count', 0)
                    })
            
            conversations.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            return conversations
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return []
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Get messages in conversation"""
        try:
            results = self.interactions.get(
                where={"conversation_id": conversation_id}
            )
            
            messages = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i]
                    
                    query = ""
                    response = ""
                    if "Query:" in doc and "Response:" in doc:
                        parts = doc.split("Response:", 1)
                        query = parts[0].replace("Query:", "").strip()
                        response = parts[1].split("Timestamp:")[0].strip()
                    
                    messages.append({
                        'query': query,
                        'response': response,
                        'raw_response': meta.get('raw_response', ''),
                        'has_raw': meta.get('has_raw', False),
                        'timestamp': meta.get('timestamp', '')
                    })
            
            messages.sort(key=lambda x: x.get('timestamp', ''))
            return messages
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return []
    
    def get_analytics(self) -> Dict:
        return {
            "total_interactions": self.interactions.count(),
            "total_conversations": self.conversations.count(),
            "model": OLLAMA_MODEL,
            "last_updated": datetime.utcnow().isoformat()
        }

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Q Chat Context Engine", version="6.3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()
session_manager = SessionManager()

active_connections: Dict[str, Dict] = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path(__file__).parent / "templates" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    return HTMLResponse(content="<h1>Q Chat V6.3.2 - Use /docs for API</h1>")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    
    active_connections[connection_id] = {
        'websocket': websocket,
        'conversation_id': None,
        'session_id': None
    }
    
    console.print(f"[cyan]🔌 Connected: {connection_id}[/cyan]")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            action = message_data.get("action", "chat")
            
            if action == "new_conversation":
                conv_id = rag.create_conversation()
                session_id = str(uuid.uuid4())
                
                active_connections[connection_id]['conversation_id'] = conv_id
                active_connections[connection_id]['session_id'] = session_id
                
                await websocket.send_json({
                    "type": "conversation_created",
                    "conversation_id": conv_id,
                    "session_id": session_id
                })
                
            elif action == "load_conversation":
                conv_id = message_data.get("conversation_id")
                messages = rag.get_conversation_messages(conv_id)
                
                session_id = str(uuid.uuid4())
                active_connections[connection_id]['conversation_id'] = conv_id
                active_connections[connection_id]['session_id'] = session_id
                
                await websocket.send_json({
                    "type": "conversation_loaded",
                    "conversation_id": conv_id,
                    "session_id": session_id,
                    "messages": messages
                })
                
            elif action == "chat":
                user_query = message_data.get("message", "")
                conv_id = active_connections[connection_id].get('conversation_id')
                session_id = active_connections[connection_id].get('session_id')
                
                if not conv_id:
                    title = user_query[:50] + "..." if len(user_query) > 50 else user_query
                    conv_id = rag.create_conversation(title)
                    session_id = str(uuid.uuid4())
                    
                    active_connections[connection_id]['conversation_id'] = conv_id
                    active_connections[connection_id]['session_id'] = session_id
                    
                    await websocket.send_json({
                        "type": "conversation_created",
                        "conversation_id": conv_id,
                        "session_id": session_id
                    })
                
                console.print(f"[blue]📨 Query: {user_query[:80]}[/blue]")
                
                # Search RAG contexts (with deduplication)
                await websocket.send_json({"type": "status", "message": "🔍 Searching context..."})
                contexts = rag.search_context(user_query, top_k=2)
                
                # Get conversation session
                session = session_manager.get_session(session_id)
                history = session.history[-2:] if session.history else []
                
                # Send context info to frontend (with timestamps)
                if contexts:
                    await websocket.send_json({
                        "type": "contexts",
                        "data": [{
                            "similarity": ctx["similarity"],
                            "query": ctx.get("query", "")[:100],
                            "response": ctx.get("response", "")[:100],
                            "timestamp": ctx.get("timestamp", "")
                        } for ctx in contexts]
                    })
                
                # Build formatted prompt
                formatted_prompt = build_single_line_prompt(user_query, contexts, history)
                
                # Send BOTH original query and formatted prompt to frontend
                await websocket.send_json({
                    "type": "formatted_prompt",
                    "original_query": user_query,  # NEW: Original query
                    "formatted_prompt": formatted_prompt,  # NEW: Full formatted prompt
                    "preview": formatted_prompt[:300] + "..." if len(formatted_prompt) > 300 else formatted_prompt,
                    "full_length": len(formatted_prompt)
                })
                
                # Execute Q Chat
                await websocket.send_json({"type": "status", "message": "🤖 Processing with Q Chat..."})
                
                result = await QChatExecutor.execute(
                    prompt=user_query,
                    contexts=contexts,
                    history=history
                )
                
                # Add to session history
                session.add_exchange(user_query, result['clean'])
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "data": result['clean'],
                    "raw_data": result['raw'],
                    "has_raw": result['has_raw']
                })
                
                # Store in vector DB (with timestamp)
                rag.store_interaction(user_query, result['clean'], result['raw'], conv_id)
                rag.update_conversation(conv_id, increment_count=True)
                
                await websocket.send_json({"type": "complete"})
            
    except WebSocketDisconnect:
        console.print(f"[yellow]🔌 Disconnected: {connection_id}[/yellow]")
        
        if connection_id in active_connections:
            session_id = active_connections[connection_id].get('session_id')
            if session_id:
                session_manager.close_session(session_id)
            
            del active_connections[connection_id]
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass

@app.get("/api/conversations")
async def get_conversations():
    conversations = rag.get_conversations()
    return JSONResponse({"conversations": conversations})

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    messages = rag.get_conversation_messages(conversation_id)
    return JSONResponse({"messages": messages})

@app.get("/api/analytics")
async def analytics():
    return JSONResponse(rag.get_analytics())

@app.delete("/api/clear")
async def clear_history():
    try:
        rag.client.delete_collection("qchat_interactions")
        rag.client.delete_collection("qchat_conversations")
        
        rag.interactions = rag.client.create_collection(
            name="qchat_interactions",
            metadata={"hnsw:space": "cosine"}
        )
        rag.conversations = rag.client.create_collection(
            name="qchat_conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        console.print("[yellow]🗑️  Cleared all history[/yellow]")
        
        return JSONResponse({"status": "success", "message": "All history cleared"})
    except Exception as e:
        console.print(f"[red]Error clearing history: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "version": "6.3.2",
        "interactions": rag.interactions.count(),
        "conversations": rag.conversations.count(),
        "active_connections": len(active_connections)
    })

if __name__ == "__main__":
    import uvicorn
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]🚀 Q Chat Context Engine V6.3.2[/bold cyan]")
    console.print("[bold white]Author: KISHORE[/bold white]")
    console.print("[bold green]✅ Timestamps + Deduplication + Enhanced View[/bold green]")
    console.print("="*80 + "\n")
    console.print(f"[green]✅ Server: http://localhost:8000[/green]")
    console.print(f"[green]✅ API Docs: http://localhost:8000/docs[/green]")
    console.print(f"[cyan]📊 Interactions: {rag.interactions.count()}[/cyan]")
    console.print(f"[cyan]💬 Conversations: {rag.conversations.count()}[/cyan]\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)