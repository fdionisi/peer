-- Tool embeddings for discovery. One row per hidden tool: name + embedding.
-- Re-indexing deletes all rows and re-inserts.
DEFINE TABLE tool SCHEMAFULL;
DEFINE FIELD name ON tool TYPE string;
DEFINE FIELD embedding ON tool TYPE option<array<float>>;

DEFINE INDEX idx_tool_embedding ON tool FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;
DEFINE INDEX idx_tool_name ON tool COLUMNS name UNIQUE;
