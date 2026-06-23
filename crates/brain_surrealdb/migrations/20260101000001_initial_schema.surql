DEFINE TABLE conversation SCHEMAFULL;
DEFINE FIELD parent ON conversation TYPE option<record<conversation>>;
DEFINE FIELD summary_content ON conversation TYPE option<string>;
DEFINE FIELD summary_created_at ON conversation TYPE option<string>;
DEFINE FIELD created_at ON conversation TYPE string;
DEFINE FIELD updated_at ON conversation TYPE string;

DEFINE TABLE message SCHEMAFULL;
DEFINE FIELD role ON message TYPE string;
DEFINE FIELD content ON message TYPE string;
DEFINE FIELD timestamp ON message TYPE string;

DEFINE TABLE conversation_item TYPE RELATION IN conversation OUT message SCHEMAFULL;
DEFINE FIELD item_type ON conversation_item TYPE string;

DEFINE TABLE current_conversation SCHEMAFULL;
DEFINE FIELD current_id ON current_conversation TYPE option<record<conversation>>;
DEFINE FIELD updated_at ON current_conversation TYPE string;

DEFINE TABLE _migrations SCHEMAFULL;
DEFINE FIELD filename ON _migrations TYPE string;
DEFINE FIELD applied_at ON _migrations TYPE string;
DEFINE INDEX idx_filename ON _migrations COLUMNS filename UNIQUE;
