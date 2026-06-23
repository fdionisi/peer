-- Embedding field on conversation — populated by the write-side of recall.
-- EMBED_DIMENSION = 1024, must agree with brain_surrealdb::EMBED_DIMENSION
-- and with the Embedder implementation (mistral-embed).
DEFINE FIELD summary_embedding ON conversation TYPE option<array<float>>;

-- HNSW vector index — enables the <|k|> KNN operator in search queries.
DEFINE INDEX idx_summary_embedding ON conversation FIELDS summary_embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;

-- General-purpose typed relation table.  Today only kind='recall' edges are
-- written; future edge kinds (entity links, etc.) use the same substrate.
DEFINE TABLE related TYPE RELATION IN conversation OUT conversation SCHEMAFULL;
DEFINE FIELD kind ON related TYPE string;
DEFINE FIELD score ON related TYPE float;
DEFINE FIELD created_at ON related TYPE string;
-- created_at is indexed so ORDER BY can use it as an idiom when reading edges
-- back in insertion order.
DEFINE INDEX idx_related_created_at ON related COLUMNS created_at;
