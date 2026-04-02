from agent_setup import get_agent
from use_rag import get_similar_texts

# 🔹 1. Get agent (system_prompt is already baked into the agent's 'prefix')
agent, system_prompt = get_agent()

# 🔹 2. Question
question = "which artist generated the most revenue"

def extract_keywords(text):
    words = text.lower().split()
    stopwords = {"what", "is", "the", "of", "a", "an", "in", "on", "for", "to", "and", "with", "who", "which", "give", "me", "info"}
    return [w for w in words if w not in stopwords]

keywords = extract_keywords(question)
print(f"\n🔑 Keywords: {keywords}")

context_hits = []
for keyword in keywords:
    try:
        results = get_similar_texts(keyword, 1)
        if results:
            context_hits.append(results)
    except Exception as e:
        print(f"RAG error: {e}")

flat_context = "\n".join([str(c) for c in context_hits])
print(f"\n📚 Retrieved Context:\n{flat_context}")
agent_input = {
    "input": f"REMEMBER THIS : {system_prompt}\n\n### RAG CONTEXT:\n{flat_context}\n\n### USER QUESTION:\n{question}"
}

print(f"\n🧠 Final Context attached. Running agent...")
agent.invoke(agent_input)
