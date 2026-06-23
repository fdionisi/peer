-- Add resolved_at to conversation, allowing branches to be marked closed.
-- NULL means the conversation is still open (main thread or unresolved branch).
DEFINE FIELD IF NOT EXISTS resolved_at ON conversation TYPE option<string>;
