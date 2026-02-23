"""
Alfred - Your Personal AI Butler with Persistent Memory

Alfred remembers everything important about you and your family across
all conversations, growing smarter and more personalized over time.

Memory backends (pick one):
  - Mem0 Cloud:  Set MEM0_API_KEY in .env  (easiest, fully managed)
  - Local:       Just set ANTHROPIC_API_KEY — uses local SQLite + Qdrant
                 with HuggingFace embeddings (no OpenAI key required)

Usage:
    python alfred.py                  # Interactive chat session
    python alfred.py --show-memories  # Display all stored memories
    python alfred.py --clear-memories # Clear all memories (use with caution)
    python alfred.py --user alice     # Run as a different user
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
import anthropic
from mem0 import Memory, MemoryClient

load_dotenv()


ALFRED_SYSTEM_PROMPT = """You are Alfred, a distinguished personal AI butler — wise, warm, and deeply attentive. \
You serve with the dedication of a trusted family confidant who has known the household for years.

Your character:
- Address the user respectfully, using their name when you know it
- Speak with quiet confidence and gentle warmth — never stiff or robotic
- You remember everything shared with you across all conversations
- When relevant memories surface, weave them naturally into your responses
- Proactively notice patterns and anticipate needs based on what you know
- Celebrate family milestones, remember preferences, and track important events
- If you recall something relevant, reference it naturally (e.g. "As I recall, your daughter Emma has a recital this week...")

Your memory purpose:
- Personal details: names, ages, relationships, occupations
- Preferences: food, hobbies, routines, communication style
- Important dates: birthdays, anniversaries, appointments
- Goals and aspirations: what the family is working toward
- Past conversations: things discussed, decisions made, follow-ups needed
- Context clues: location, schedule patterns, recurring topics

Always be helpful, discreet, and genuinely interested in the wellbeing of the household."""


def _local_mem0_config(anthropic_api_key: str) -> dict:
    """
    Build a Mem0 config that uses:
      - Anthropic (Claude) as the LLM for memory extraction
      - HuggingFace sentence-transformers for local embeddings (no OpenAI key needed)
      - Qdrant (on-disk) as the vector store
      - SQLite for history
    """
    qdrant_path = os.path.expanduser("~/.alfred/qdrant")
    db_path = os.path.expanduser("~/.alfred/alfred_memory.db")
    os.makedirs(qdrant_path, exist_ok=True)

    return {
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": "claude-haiku-4-5-20251001",
                "api_key": anthropic_api_key,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                # Small, fast model that runs locally — no API key required
                "model": "multi-qa-MiniLM-L6-cos-v1",
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "alfred_memories",
                "on_disk": True,
                "path": qdrant_path,
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": db_path,
        "version": "v1.1",
    }


def get_memory_client(anthropic_api_key: str | None = None) -> Memory | MemoryClient:
    """
    Return the appropriate Mem0 client.

    Priority:
      1. MEM0_API_KEY set → use Mem0 Cloud (MemoryClient)
      2. Otherwise        → use local Memory with Anthropic LLM + HuggingFace embeddings
    """
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if mem0_api_key:
        print("  [Alfred: using Mem0 Cloud for persistent memory]")
        return MemoryClient(api_key=mem0_api_key)

    key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set to use local memory. "
            "Copy .env.example to .env and fill in your key."
        )
    print("  [Alfred: using local memory (Qdrant + SQLite + HuggingFace embeddings)]")
    config = _local_mem0_config(key)
    return Memory.from_config(config)


def format_memories(memories: list[dict]) -> str:
    """Format a list of memory dicts into a readable context block."""
    if not memories:
        return ""
    lines = ["[Relevant memories about this household:]"]
    for m in memories:
        text = m.get("memory", m.get("text", ""))
        if text:
            lines.append(f"  • {text}")
    return "\n".join(lines)


def retrieve_memories(mem: Memory | MemoryClient, query: str, user_id: str) -> str:
    """Search Mem0 for memories relevant to the current query."""
    try:
        results = mem.search(query=query, user_id=user_id, limit=10)
        items = results.get("results", []) if isinstance(results, dict) else (results or [])
        return format_memories(items)
    except Exception as exc:
        print(f"  [Alfred memory note: retrieval issue — {exc}]", file=sys.stderr)
        return ""


def store_memories(mem: Memory | MemoryClient, messages: list[dict], user_id: str) -> None:
    """Extract and store important facts from the latest exchange."""
    try:
        mem.add(messages=messages, user_id=user_id)
    except Exception as exc:
        print(f"  [Alfred memory note: storage issue — {exc}]", file=sys.stderr)


def show_all_memories(mem: Memory | MemoryClient, user_id: str) -> None:
    """Print all stored memories for a user."""
    try:
        all_memories = mem.get_all(user_id=user_id)
        items = (
            all_memories.get("results", [])
            if isinstance(all_memories, dict)
            else (all_memories or [])
        )

        if not items:
            print(f"\nAlfred has no stored memories yet for user '{user_id}'.")
            return

        print(f"\n{'='*60}")
        print(f"  Alfred's Memory Bank  ({len(items)} memories)")
        print(f"  User: {user_id}")
        print(f"{'='*60}")
        for i, m in enumerate(items, 1):
            text = m.get("memory", m.get("text", ""))
            created = m.get("created_at", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    created = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
            print(f"  {i:3}. {text}")
            if created:
                print(f"       Stored: {created}")
        print(f"{'='*60}\n")
    except Exception as exc:
        print(f"Could not retrieve memories: {exc}")


def clear_all_memories(mem: Memory | MemoryClient, user_id: str) -> None:
    """Delete all memories for a user after confirmation."""
    confirm = input(
        f"Delete ALL memories for user '{user_id}'? This cannot be undone. (yes/no): "
    )
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return
    try:
        mem.delete_all(user_id=user_id)
        print(f"All memories for '{user_id}' have been cleared.")
    except Exception as exc:
        print(f"Could not clear memories: {exc}")


def chat(user_id: str) -> None:
    """
    Run an interactive Alfred chat session with persistent memory.

    Each exchange:
      1. Retrieves relevant memories for context
      2. Sends memory context + conversation history to Claude
      3. Stores important facts from the exchange into Mem0
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your Anthropic API key."
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=anthropic_key)
    mem = get_memory_client(anthropic_key)

    # Conversation history for the current session (Claude message format)
    conversation: list[dict] = []

    print("\n" + "=" * 60)
    print("  Alfred — Your Personal AI Butler")
    print("  Powered by Claude + Mem0 persistent memory")
    print(f"  Serving: {user_id}")
    print("  Type 'quit' or 'exit' to end the session")
    print("  Type 'memories' to see what Alfred remembers")
    print("=" * 60 + "\n")
    print("Alfred: Good day. How may I be of service?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAlfred: Until next time. It has been a pleasure.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nAlfred: Very good. I shall be here whenever you need me.")
            break

        if user_input.lower() == "memories":
            show_all_memories(mem, user_id)
            continue

        # 1. Retrieve relevant memories to give Alfred context
        relevant_memories = retrieve_memories(mem, user_input, user_id)

        # 2. Build the system prompt, injecting retrieved memories
        system_with_memory = ALFRED_SYSTEM_PROMPT
        if relevant_memories:
            system_with_memory = ALFRED_SYSTEM_PROMPT + "\n\n" + relevant_memories

        # 3. Add user message to conversation history
        conversation.append({"role": "user", "content": user_input})

        # 4. Call Claude with the enriched system prompt
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_with_memory,
                messages=conversation,
            )
            assistant_reply = response.content[0].text
        except anthropic.APIError as exc:
            print(f"\nAlfred: I do beg your pardon — I encountered a difficulty: {exc}\n")
            conversation.pop()
            continue

        # 5. Add Alfred's reply to conversation history
        conversation.append({"role": "assistant", "content": assistant_reply})
        print(f"\nAlfred: {assistant_reply}\n")

        # 6. Store this exchange in Mem0 for future sessions
        store_memories(mem, conversation[-2:], user_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alfred — Your Personal AI Butler with Persistent Memory"
    )
    parser.add_argument(
        "--user",
        default=os.getenv("ALFRED_USER_ID", "default_user"),
        help="User ID for memory scoping (default: ALFRED_USER_ID env var or 'default_user')",
    )
    parser.add_argument(
        "--show-memories",
        action="store_true",
        help="Display all memories Alfred has stored for this user",
    )
    parser.add_argument(
        "--clear-memories",
        action="store_true",
        help="Clear all stored memories for this user (irreversible)",
    )
    args = parser.parse_args()

    if args.show_memories or args.clear_memories:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        mem = get_memory_client(anthropic_key)
        if args.show_memories:
            show_all_memories(mem, args.user)
        if args.clear_memories:
            clear_all_memories(mem, args.user)
        return

    chat(user_id=args.user)


if __name__ == "__main__":
    main()
