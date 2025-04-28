#!/usr/bin/env python
import uvicorn
from fastapi import FastAPI, HTTPException # Import HTTPException for errors
from pydantic import BaseModel
from typing import Dict, Any, Literal, List, Optional # Add List, Optional
from fastapi.middleware.cors import CORSMiddleware # For allowing frontend requests
from fastapi.responses import FileResponse # Import FileResponse
from fastapi.staticfiles import StaticFiles # Import StaticFiles
import os

# Import the core interaction logic from our refactored runner script
# Ensure loop_agent_runner.py is in the same directory or accessible via PYTHONPATH
try:
    from loop_agent_runner import run_comment_interaction
except ImportError as e:
    print(f"Error importing from loop_agent_runner: {e}")
    print("Make sure loop_agent_runner.py is in the correct path.")
    exit(1)

# --- Mock Blog Post Data ---
# Using slugs as keys for easy URL mapping
mock_posts = {
    "tech-trends-2024": {
        "title": "The plantation of our ancestors",
        "slug": "tech-trends-2024",
        "content": "Maartje Duin follows the discussion about white privilege with growing discomfort. Her mother is a baroness from an old noble family; what role did her ancestors play in the Dutch slavery past? Maartje dives into her family tree and discovers that her great-great-great-grandmother was co-owner of a Surinamese plantation when slavery was abolished in 1863. Among the freed slaves, one surname stands out: Bouva. Are their descendants still alive?",
        "audio_url": "/static/sprint3-podcast.mp3",
        "comments": [
            {"id": "c1", "commenter": "AI_fan", "text": "Great overview! GenAI is changing everything.", "agent_response": None},
            {"id": "c2", "commenter": "Skeptic", "text": "Quantum feels overhyped. What are the real use cases?", "agent_response": None}
        ]
    },
    "mindful-coding": {
        "title": "The comeback of Mr Red Flag, Marilyn Manson",
        "slug": "mindful-coding",
        "content": "We stood with mixed feelings yesterday in the AFAS Live, where Marilyn Manson played his first Dutch show since he was accused of sexual abuse and assault by a dozen women. Incidentally, there was no sign of that discomfort in the audience, because I think we were the only ones last night who viewed the anti-Christ superstar through those glasses. About the comeback of Mr Red Flags and other controversial artists. And further: Leon Ramakers, one of the founders of the Dutch live sector, talks about his latest project: Cream radio.",
        "audio_url": "/static/2.mp3",
        "comments": [
             {"id": "c3", "commenter": "DevDude", "text": "Interesting concept, but hard to implement with deadlines.", "agent_response": None},
        ]
    },
    "climate-action-now": {
        "title": "Away with X and Meta",
        "slug": "climate-action-now",
        "content": "Since the inauguration of Donald Trump on January 20th and the aggressive grabbing and creeping of Elon Musk in the American government institutions, a seismic shift has occurred among progressive citizens and cultural institutions when it comes to social media. Do we still have to be in that pool of perdition called X, where supposedly free speech reigns but where the algorithms are clearly set in such a way that the far-right filth sloshes over the edges? And what should we do with Meta, which also licks Trump's boots? And also in this episode: topliners, the invisible sharpshooters of the songwriting world.",
        "audio_url": "/static/3.mp3",
        "comments": []
    }
}
next_comment_id = 4 # Simple counter for new comment IDs

# --- API Data Models ---

# Model for the agent response part within a comment
class AgentResponse(BaseModel):
    tone: str
    text: str

# Model for a single comment
class Comment(BaseModel):
    id: str
    commenter: str # Keep simple for now, could be more complex
    text: str
    agent_response: Optional[AgentResponse] = None # Agent response can be None initially

# Model for returning a list of posts (summary)
class PostSummary(BaseModel):
    title: str
    slug: str

# Model for returning detailed post info including comments
class PostDetail(BaseModel):
    title: str
    slug: str
    content: str
    comments: List[Comment]
    audio_url: Optional[str] = None

# Model for adding a new comment
class NewCommentRequest(BaseModel):
    commenter: str = "Anonymous" # Default commenter
    text: str
    tone: Literal["neutral", "strict", "aggressive", "optimistic", "happy", "humorous"]

# --- FastAPI App Setup ---

app = FastAPI(
    title="Mock Blog Comment Agent API",
    description="API for a mock blog site where agents respond to comments."
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Add CORS middleware to allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---

@app.get("/", include_in_schema=False) # Add endpoint for root path
async def read_index():
    """Serves the index.html file for the frontend."""
    return FileResponse('index.html')

# List all available blog posts (summaries)
@app.get("/api/posts", response_model=List[PostSummary])
async def list_posts():
    return [PostSummary(title=data["title"], slug=data["slug"]) for data in mock_posts.values()]

# Get details for a specific blog post
@app.get("/api/posts/{post_slug}", response_model=PostDetail)
async def get_post(post_slug: str):
    post = mock_posts.get(post_slug)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return PostDetail(**post)

# Add a comment to a specific blog post and trigger agent response
@app.post("/api/posts/{post_slug}/comments", response_model=Comment)
async def add_comment(post_slug: str, request: NewCommentRequest):
    global next_comment_id
    post = mock_posts.get(post_slug)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    # Prepare context for the agent
    post_context = f"Blog Post Title: {post['title']}\n\n{post['content']}"

    # Run the agent interaction logic
    print(f"Running agent for new comment on '{post_slug}' with tone '{request.tone}'")
    agent_result = run_comment_interaction(
        comment=request.text,
        context=post_context,
        tone=request.tone
    )

    generated_response_text = None
    if agent_result.get("error"):
        print(f"Agent Error: {agent_result['error']}")
        # Decide how to handle agent errors - maybe store comment without response?
        # For now, we'll store it without an agent response.
        agent_response_obj = None
    elif agent_result.get("final_response"):
        generated_response_text = agent_result["final_response"]
        agent_response_obj = AgentResponse(tone=request.tone, text=generated_response_text)
        print(f"Agent Response Generated: {generated_response_text[:100]}...")
    else:
        print("Agent did not produce a final response.")
        agent_response_obj = None


    # Create and store the new comment (including agent response if generated)
    new_comment_data = Comment(
        id=f"c{next_comment_id}",
        commenter=request.commenter,
        text=request.text,
        agent_response=agent_response_obj
    )
    mock_posts[post_slug]["comments"].append(new_comment_data.model_dump()) # Store as dict
    next_comment_id += 1

    return new_comment_data # Return the newly added comment object

# --- Server Execution (for direct running) ---

if __name__ == "__main__":
    print("Starting Uvicorn server for Mock Blog...")
    # Run the server on host 0.0.0.0 to make it accessible on the network
    # Default port is 8000
    # Use reload=True for development to automatically restart on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True) 