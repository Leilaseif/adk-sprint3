#!/usr/bin/env python
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs a sequence of agents to analyze and respond to a comment within a given context and tone.

import json
import uuid # Import uuid for unique session IDs
from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from dotenv import load_dotenv

from .util import load_instruction_from_file

# --- Agent Definitions ---
# (Keep agent definitions: state_setup_agent, comment_analyzer_agent, comment_responder_agent)
state_setup_agent = LlmAgent(
    name="StateSetup",
    model="gemini-1.5-pro-latest", # Use Pro for reliable JSON parsing/output
    instruction=load_instruction_from_file("state_setup_instruction.txt"),
    output_key="initial_data" # Output the dict containing comment, context, tone
)
comment_analyzer_agent = LlmAgent(
    name="CommentAnalyzer",
    model="gemini-1.5-flash",
    instruction=load_instruction_from_file("comment_analyzer_instruction.txt"),
    output_key="analysis",
)
comment_responder_agent = LlmAgent(
    name="CommentResponder",
    model="gemini-1.5-flash",
    instruction=load_instruction_from_file("comment_responder_instruction.txt"),
    output_key="final_response",
)

# --- Sequence Agent Workflow ---
comment_interaction_agent = SequentialAgent(
    name="CommentInteraction",
    sub_agents=[state_setup_agent, comment_analyzer_agent, comment_responder_agent]
)

# --- Global/Shared Resources ---
load_dotenv() # Load .env once
APP_NAME = "comment_responder_app"
USER_ID = "web_user_01" # Generic user ID for web interface
session_service = InMemorySessionService() # Instantiate session service globally


# --- Interaction Function (Refactored) ---
def run_comment_interaction(comment: str, context: str, tone: str) -> dict:
    session_id = f"session_{uuid.uuid4()}" # Generate unique session ID
    print(f"Starting interaction for session: {session_id}")

    # Prepare initial message
    initial_message_data = {"comment": comment, "context": context, "tone": tone}
    initial_message_content = json.dumps(initial_message_data)
    initial_message = types.Content(role="user", parts=[types.Part(text=initial_message_content)])

    # Explicitly create the session
    try:
        session = session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except Exception as e:
        print(f"Error creating session {session_id}: {e}")
        return {"error": f"Failed to create session: {e}"}

    # Create a Runner
    runner = Runner(
        agent=comment_interaction_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    final_response_text = None
    error_message = None
    final_state_dict = {}

    try:
        print(f"--- Running sequence for session: {session_id} ---")
        events = runner.run(
            user_id=USER_ID,
            session_id=session_id,
            new_message=initial_message
        )

        # Process events
        for event in events:
            if event.is_final_response():
                final_response_text = event.content.parts[0].text

        # Fetch final state
        final_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        if final_session:
            final_state_dict = final_session.state
            print(f"Final State for {session_id}: {final_state_dict}") # Log final state
        else:
             print(f"Warning: Could not retrieve final state for session {session_id}")
             error_message = "Failed to retrieve final session state."

    except Exception as e:
        print(f"Error during agent execution for session {session_id}: {e}")
        # Attempt to fetch state even after error, might have partial results
        try:
            final_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            if final_session:
                final_state_dict = final_session.state
        except Exception as final_state_e:
             print(f"Could not retrieve state after error for {session_id}: {final_state_e}")
        error_message = f"Agent execution failed: {e}"

    # --- Prepare results --- 
    result = {
        "session_id": session_id,
        "final_state": final_state_dict,
        "final_response": final_state_dict.get("final_response"), # Get from state if possible
        "error": error_message
    }

    # Fallback to response text from event if not in state
    if not result["final_response"] and final_response_text:
        result["final_response"] = final_response_text

    # Clean up session from memory if needed (optional, depends on expected load)
    # session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    print(f"Interaction complete for session: {session_id}")
    return result

# --- Remove Example Usage Block --- 
# (The old example usage calling run_comment_interaction directly is removed)

# --- Add basic main guard if running this file directly isn't intended ---
# if __name__ == "__main__":
#    print("This script is intended to be imported by a web server (e.g., main.py), not run directly.")
