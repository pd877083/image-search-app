# Demo Script: Semantic Image Search

This script is designed for a five-minute project demonstration. It uses the **Room & Spaceship** analogy to explain cross-modal retrieval without assuming that the audience already knows CLIP.

## The core idea: the Room & Spaceship analogy

Imagine a very large room in space. Every image and every sentence is placed somewhere in that room.

- A picture of a dog running on a beach and the sentence *"a dog running on the beach"* are placed close together.
- A picture of a rocket and the sentence *"a spaceship flying in the sky"* are also close together, but in a different area of the room.
- Unrelated things, such as a bicycle and a kitchen, are far apart.

CLIP is the navigator that knows how to place both images and text in this shared room. When a user types a query, we turn it into a point in the room and ask: **which image-points are nearest?** The app returns those nearby images.

The "spaceship" part is the search: rather than opening every image one by one, our spaceship flies directly to the closest neighbourhood in the room. This is semantic search: it can match meaning, not only identical words. For example, a query such as *"child playing in water"* can retrieve an image captioned with different wording.

Technically, the room is a shared **512-dimensional embedding space**. The app uses cosine similarity to measure which stored vectors are closest to the query vector.

## Five-minute walkthrough

### 0:00–0:35 — Problem and solution

“Finding an image in a large collection normally depends on filenames or exact keywords. That fails when the user describes the same idea in different words. Our project is a semantic image-search system: it accepts text or an image and retrieves the most relevant items from the Flickr8k gallery.”

“The key technology is a vision-language model. It gives images and language a shared representation, so we can compare them directly.”

### 0:35–1:15 — Explain the analogy

“Think of the embedding space as a giant room. CLIP puts matching images and sentences into the same neighbourhood. A beach-dog image is near the idea of ‘dog running on the beach’; a rocket image is near ‘spaceship in the sky.’ At search time, we convert the query into a point in this room and return the nearest image-points.”

“The important point is that we are retrieving from existing images; this application does not generate new images.”

### 1:15–2:10 — Text-to-image search

1. Open **Text → Images**.
2. Enter a natural-language query, for example: `a dog running on the beach`.
3. Choose the number of results and select **Search Images**.
4. Point out the ranked results and their similarity scores.

Say: “The text query is encoded once by our fine-tuned CLIP model. We compare it against precomputed embeddings for the gallery images and show the Top-K closest results. Because the vectors are L2-normalized, their inner product is equivalent to cosine similarity.”

Optional: repeat with a paraphrase such as `a puppy playing by the sea` to demonstrate semantic matching. Point out the recent-search buttons if they are visible.

### 2:10–2:50 — Image-to-caption search

1. Open **Image → Captions**.
2. Upload one image from the gallery or a similar photograph.
3. Select **Find Captions**.

Say: “Here the direction is reversed. The image is placed in the same room, then matched against precomputed caption embeddings. This shows that the system is genuinely cross-modal: the query can be an image while the results are text.”

### 2:50–3:30 — Why semantic search helps

1. Open **CLIP vs TF-IDF vs BM25**.
2. Use the same natural-language query and select **Compare Methods**.

Say: “TF-IDF and BM25 are strong traditional text-retrieval baselines, but they primarily rely on word overlap. CLIP compares meanings in the shared room, so it can still work when the query uses synonyms or different phrasing. This tab makes that trade-off visible rather than claiming it without a baseline.”

### 3:30–4:15 — Model and performance comparison

1. Open **CLIP vs BLIP vs ALIGN**.
2. Run a query and compare the returned images and Hit@K chart.

Say: “We also compare three vision-language models: CLIP, BLIP, and ALIGN. On the project’s ground-truth evaluation, the displayed Hit@K scores show retrieval quality at different cutoffs. This separates model quality from user-interface impressions.”

Optional: return to the text-search tab, turn on **ONNX mode**, and run the same query. Say: “ONNX is an inference format. The INT8 option trades a small amount of numerical precision for a smaller, faster CPU model; it does not change the stored gallery or the retrieval logic.”

### 4:15–5:00 — Architecture, result, and limitations

“For efficiency, gallery images and captions are embedded offline once and saved to disk. At runtime, only the user’s query is encoded. FAISS performs exact Top-K inner-product search over normalized vectors; if FAISS is unavailable, the app safely falls back to NumPy.”

“We fine-tuned the CLIP checkpoint on image-caption pairs, then used the same checkpoint to build the gallery and encode queries. Using a different checkpoint for either side would put them in incompatible coordinate systems and hurt retrieval.”

“The result is a practical semantic-search application with text-to-image, image-to-text, image-to-image, baseline comparisons, model comparisons, and an optional optimized inference path. Its limits are the size and bias of Flickr8k, ambiguous queries, and the fact that ranking is only as good as the learned representations and gallery coverage.”

## Likely viva questions and answers

### 1. What is semantic image search?

It retrieves images using their meaning rather than only filenames, tags, or exact keywords. A text query and an image can be compared because both are represented as vectors in the same embedding space.

### 2. What is CLIP, and why did you use it?

CLIP is a vision-language model trained to align images with their related text. We use its ViT-B/32 backbone because it can encode both modalities into a common 512-dimensional space, which makes cross-modal retrieval possible.

### 3. Explain the Room & Spaceship analogy in technical terms.

The room is the shared embedding space. Every image and text input becomes a vector—a coordinate in that space. The spaceship is the nearest-neighbour search process: it starts at the query vector and returns stored vectors with the highest similarity.

### 4. How does a text query return images?

The query is tokenized and encoded into a normalized text embedding. We compare it with precomputed, normalized image embeddings and rank them by cosine similarity. The highest-scoring image paths are displayed as the Top-K results.

### 5. Why do you precompute embeddings instead of encoding every image for each search?

The gallery is mostly static. Encoding it once avoids repeating expensive model inference for thousands of images on every request. At query time, the system only encodes one input and performs fast vector ranking.

### 6. Why use cosine similarity and L2 normalization?

Cosine similarity compares vector direction, which represents semantic alignment, rather than vector magnitude. After L2 normalization, cosine similarity equals the dot product, so FAISS can use an efficient inner-product index while preserving the intended ranking.

### 7. What is FAISS, and what happens if it is unavailable?

FAISS is a vector-search library used here for exact Top-K inner-product retrieval with `IndexFlatIP`. The application includes a NumPy dot-product fallback, so it remains functional without FAISS and returns the same ranking logic.

### 8. Why must the query encoder and stored gallery embeddings use the same checkpoint?

An embedding only has meaning relative to the model that produced it. If the gallery is encoded with one set of fine-tuned weights and the query with another, their coordinates are no longer aligned; similarity scores can become unreliable. Therefore the app uses the same preferred fine-tuned CLIP checkpoint for both.

### 9. How is this different from TF-IDF or BM25?

TF-IDF and BM25 are lexical retrieval methods that use term occurrence and overlap. They are useful baselines, but they do not directly understand visual content or cross-modal similarity. CLIP can retrieve a relevant image when the query is a paraphrase or uses synonyms, although it can still fail on ambiguous or uncommon concepts.

### 10. What are the limitations and possible improvements?

The gallery is limited to Flickr8k, so coverage, caption quality, and dataset bias affect results. The model may struggle with fine-grained attributes, counting, negation, and vague queries. Improvements include a larger and more diverse dataset, relevance feedback, metadata filters, hard-negative fine-tuning, approximate indexing for much larger galleries, and a more rigorous evaluation set.

## One-sentence closing

“Our system turns images and language into locations in one shared semantic room, then uses fast vector search to navigate from a user’s idea to the most relevant images or captions.”
