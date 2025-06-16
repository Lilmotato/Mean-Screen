from app.services.qdrant_client import init_policy_collection, add_policy, search_similar_policies

init_policy_collection()

text = "Posts attacking individuals based on race are not allowed"
metadata = {"provider": "reddit", "type": "content_moderation"}
add_policy(text, metadata)

results = search_similar_policies("I dislike people of this race")
print("Top matches:")
for r in results:
    print(r.payload["text"], "→", r.score)
