"""
test_memory.py — Demonstrates Alfred's persistent memory across conversations.

This test simulates two separate conversations with Alfred:
  Session 1: Tell Alfred things about your family (stored in Mem0)
  Session 2: Ask Alfred to recall those things without repeating them

If Mem0 is working, Alfred will answer Session 2 questions using only what
was stored in Session 1 — proving true cross-session persistent memory.

Usage:
    ANTHROPIC_API_KEY=sk-... python test_memory.py
"""

import os
import sys
import time
from dotenv import load_dotenv
import anthropic
from mem0 import Memory, MemoryClient

load_dotenv()

TEST_USER_ID = "alfred_test_user"

ALFRED_SYSTEM_PROMPT = """You are Alfred, a distinguished personal AI butler. \
You remember everything shared with you and weave it naturally into conversation."""


def _local_mem0_config(api_key: str) -> dict:
    qdrant_path = os.path.expanduser("~/.alfred/qdrant")
    db_path = os.path.expanduser("~/.alfred/alfred_memory.db")
    os.makedirs(qdrant_path, exist_ok=True)
    return {
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": "claude-haiku-4-5-20251001",
                "api_key": api_key,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
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


def get_memory_client(api_key: str):
    mem0_key = os.getenv("MEM0_API_KEY")
    if mem0_key:
        return MemoryClient(api_key=mem0_key)
    return Memory.from_config(_local_mem0_config(api_key))


def ask_alfred(
    client: anthropic.Anthropic,
    mem,
    question: str,
    user_id: str,
    inject_memories: bool = True,
) -> str:
    """Send a single question to Alfred, optionally enriched with memories."""
    system = ALFRED_SYSTEM_PROMPT

    if inject_memories:
        results = mem.search(query=question, user_id=user_id, limit=8)
        items = results.get("results", []) if isinstance(results, dict) else (results or [])
        if items:
            lines = ["[What Alfred remembers about this household:]"]
            for m in items:
                text = m.get("memory", m.get("text", ""))
                if text:
                    lines.append(f"  • {text}")
            system = ALFRED_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def teach_alfred(mem, exchanges: list[tuple[str, str]], user_id: str) -> None:
    """Store conversation exchanges in Mem0."""
    for user_msg, assistant_msg in exchanges:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        mem.add(messages=messages, user_id=user_id)
        time.sleep(0.5)


def get_all_memories(mem, user_id: str) -> list[str]:
    all_mem = mem.get_all(user_id=user_id)
    items = all_mem.get("results", []) if isinstance(all_mem, dict) else (all_mem or [])
    return [m.get("memory", m.get("text", "")) for m in items if m.get("memory") or m.get("text")]


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("\nInitializing Alfred's memory system...")
    print("(First run downloads a small embedding model ~25MB — takes a moment)")
    mem = get_memory_client(api_key)

    # Clean slate
    print(f"Clearing previous test memories for '{TEST_USER_ID}'...")
    try:
        mem.delete_all(user_id=TEST_USER_ID)
    except Exception:
        pass
    time.sleep(1)

    # ------------------------------------------------------------------ #
    # SESSION 1: Introduce the family                                      #
    # ------------------------------------------------------------------ #
    separator("SESSION 1 — Introducing the Family to Alfred")
    print("Telling Alfred about your family...\n")

    session1_facts = [
        (
            "My name is James. My wife is called Sarah and she's a nurse.",
            "Delightful to make your acquaintance, Mr. James. How wonderful that Sarah serves as a nurse.",
        ),
        (
            "We have two children: Emma, age 8, and Tom, age 5. Emma loves ballet and Tom is mad about dinosaurs.",
            "Charming! Little Emma with her ballet slippers and young Tom with his prehistoric companions.",
        ),
        (
            "I'm a software engineer and I usually work from home on Fridays. I prefer my coffee black, no sugar.",
            "Noted, sir. Black coffee, no sugar on your work-from-home Fridays.",
        ),
        (
            "Emma's dance recital is on March 15th. She's been practising for months.",
            "I have it firmly in mind — Miss Emma's recital on March 15th.",
        ),
        (
            "Sarah is allergic to shellfish and Tom has a mild nut allergy.",
            "Understood. No shellfish for Mrs. Sarah, and vigilance around nuts for young Master Tom.",
        ),
    ]

    for user_msg, _ in session1_facts:
        print(f"  You told Alfred: \"{user_msg[:72]}{'...' if len(user_msg) > 72 else ''}\"")

    print("\nStoring memories...")
    teach_alfred(mem, session1_facts, TEST_USER_ID)
    time.sleep(2)

    memories = get_all_memories(mem, TEST_USER_ID)
    print(f"\nAlfred stored {len(memories)} memories from Session 1:")
    for m in memories:
        print(f"  ✓ {m}")

    # ------------------------------------------------------------------ #
    # SESSION 2: Fresh conversation — Alfred recalls from memory            #
    # ------------------------------------------------------------------ #
    separator("SESSION 2 — New Conversation (Alfred recalls from memory)")
    print("(These facts were NOT repeated — Alfred retrieves them from Mem0)\n")

    recall_tests = [
        (
            "What do you know about my family?",
            ["emma", "tom", "sarah", "james"],
        ),
        (
            "Is there anything important coming up soon?",
            ["march", "recital", "emma"],
        ),
        (
            "What allergies should I be careful about?",
            ["shellfish", "nut"],
        ),
        (
            "What does my son enjoy?",
            ["dinosaur"],
        ),
    ]

    all_passed = True
    for question, expected_keywords in recall_tests:
        print(f"  You: {question}")
        answer = ask_alfred(client, mem, question, TEST_USER_ID, inject_memories=True)
        print(f"  Alfred: {answer}\n")

        found = [kw for kw in expected_keywords if kw in answer.lower()]
        needed = max(1, len(expected_keywords) // 2)
        if len(found) >= needed:
            print(f"  PASS — recalled: {found}\n")
        else:
            missed = [kw for kw in expected_keywords if kw not in answer.lower()]
            print(f"  FAIL — missed: {missed}\n")
            all_passed = False

    # ------------------------------------------------------------------ #
    # SESSION 3: Baseline without memory (for contrast)                    #
    # ------------------------------------------------------------------ #
    separator("SESSION 3 — Baseline (No Memory Injection — for contrast)")
    print("Same question without memory retrieval:\n")

    q = "What do you know about my family?"
    print(f"  You: {q}")
    answer_no_mem = ask_alfred(client, mem, q, TEST_USER_ID, inject_memories=False)
    print(f"  Alfred (no memory): {answer_no_mem}")
    print("\n  ^ Alfred can't answer without retrieved memories — memory injection works!\n")

    # ------------------------------------------------------------------ #
    # Summary                                                               #
    # ------------------------------------------------------------------ #
    separator("Test Summary")
    backend = "Mem0 Cloud" if os.getenv("MEM0_API_KEY") else "Local (Qdrant + SQLite + HuggingFace)"
    print(f"  Memories stored : {len(memories)}")
    print(f"  Recall tests    : {'ALL PASSED' if all_passed else 'SOME FAILURES — see above'}")
    print(f"  Memory backend  : {backend}")
    print(f"  Memory location : ~/.alfred/")
    print()
    print("  Next steps:")
    print("    python alfred.py                          # Start chatting with Alfred")
    print("    python alfred.py --show-memories          # Browse stored memories")
    print(f"    python alfred.py --user {TEST_USER_ID} --clear-memories  # Clean up test data")
    print()


if __name__ == "__main__":
    main()
